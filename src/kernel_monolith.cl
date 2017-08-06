#include "geom.h"
#include "random.cl"
#include "intersect.cl"
#include "utils.cl"
#include "bvh.cl"
#include "env_map.cl"

// x and y include offsets when supersampling
inline Ray getCameraRay(const float2 pos, global RenderParams *params)
{
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction

    // NDC-space, [0,1]x[0,1]
    float NDCx = pos.x / params->width;
    float NDCy = pos.y / params->height;

    // Screen space, [-1,1]x[-1,1]
    float SCRx = 2.0f * NDCx - 1.0f;
    float SCRy = 2.0f * NDCy - 1.0f;

    // Aspect ratio fix applied horizontally
    SCRx *= (float)params->width / params->height;

    // Screen space coordinates scaled based on fov
    float scale = tan(toRad(0.5f * params->camera.fov)); // half of width
    SCRx *= scale;
    SCRy *= scale;

    // World space coorinates of pixel
    float3 rayTarget = params->camera.pos + params->camera.right * SCRx + params->camera.up * SCRy + params->camera.dir;
    float3 rayDirection = normalize(rayTarget - params->camera.pos);

    // Construct camera ray
    Ray r = { params->camera.pos, rayDirection };
    return r;
}

inline Hit raycast(Ray *r, float tMax, global Triangle *tris, global GPUNode *nodes, global uint *indices, global RenderParams *params, bool sampleImplicit)
{
    Hit hit = EMPTY_HIT(tMax);
    bvh_intersect(r, &hit, tris, nodes, indices);

    if (sampleImplicit)
        intersectLight(&hit, r, params);

    return hit;
}

// Blinn-Phong
inline float3 calcLighting(PointLight light, float3 V, Hit *hit, global Material *materials, global uchar *texData, global TexDescriptor *textures, global Triangle *tris, global GPUNode *nodes, global uint *indices, global RenderParams *params)
{
    float3 L = light.pos - hit->P;
    float dist = length(L);
    L = normalize(L);

    Ray shadowRay = { (hit->P + 1e-3f * hit->N), L };
    Hit shdw = raycast(&shadowRay, dist, tris, nodes, indices, params, false);
    float visibility = (shdw.i == -1) ? 1.0f : 0.0f; // early exits useless on GPU

    // Testing material:
    float3 Ks = (float3)(1.0f); // fraction of light reflected, per color channel
    float glossiness = 0.025f; // probably not the right name...

    float3 H = normalize(L + V);
    float3 diffuse = materials[hit->i].Kd * max(0.0f, dot(L, hit->N));
    float3 specular = Ks * pow(max(0.0f, dot(hit->N, H)), 1.0f / glossiness);

    if(dot(hit->N, L) < 0) specular = (float3)(0.0f);

    float falloff = 1.0f / (dist * dist + 1e-5f);
    return visibility * light.E * falloff * (diffuse + specular);
}

inline float3 whittedShading(Hit *hit, global Material *materials, global uchar *texData, global TexDescriptor *textures, global Triangle *tris, global GPUNode *nodes, global uint *indices, global PointLight *lights, global RenderParams *params)
{
    float3 V = normalize(params->camera.pos - hit->P); // P to eye
    float vDotN = dot(V, hit->N);
    if(vDotN < 0)
    {
        hit->N *= -1.0f;
    }

    float3 res = (float3)(0.0f);

    // Point light assumed for now
    for(uint i = 0; i < params->n_lights; i++)
    {
        PointLight light = lights[i];
        res += calcLighting(light, V, hit, materials, texData, textures, tris, nodes, indices, params);
    }

    // Optional light at camera pos
    if(params->flashlight)
    {
        float3 lPos = params->camera.pos + 0.1f * params->camera.dir;
        PointLight flashlight = { (float3)(10.0f), lPos };
        res += calcLighting(flashlight, V, hit, materials, texData, textures, tris, nodes, indices, params);
    }

    return res;
}

inline float3 reflectionShading(Hit *hit, read_only image2d_t envMap, global RenderParams *params)
{
    float3 V = normalize(params->camera.pos - hit->P); // P to eye
    float vDotN = dot(V, hit->N);
    if(vDotN < 0)
    {
        hit->N *= -1.0f;
    }

    return evalEnvMapDir(envMap, reflect(-V, hit->N)) - (float3)(0.1f);
}

// Ray tracing!
inline float3 traceRay(float2 pos, global uchar *texData, global TexDescriptor *textures, global PointLight *lights, global Triangle *tris, global Material *materials, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params, global RenderStats *stats)
{
    float3 pixelColor = (float3)(0.0f);

    // Supersampling
    const int SAMPLES = 1;
    const int dim = (int)sqrt((float)SAMPLES);
    for (int n = 0; n < SAMPLES; n++)
    {
        pos += sampleRegular(n, dim);
        Ray r = getCameraRay(pos, params);
        Hit hit = raycast(&r, FLT_MAX, tris, nodes, indices, params, false);
        atomic_inc(&stats->primaryRays);

        if (hit.i > -1)
        {
            // Whitted shading
            //pixelColor = whittedShading(&hit, materials, texData, textures, tris, nodes, indices, lights, params);

            // Reflections + environment map
            //pixelColor += reflectionShading(&hit, envMap, params);

            // Depth shading
            // pixelColor = (float3)(hit.t / 8.0f);

            // Intersection shading
            pixelColor = materials[hit.matId].Kd;

            //pixelColor = (float3)(1.0f, 0.0f, 0.0f) / (2.0f * hit.t);
        }
        else if (params->useEnvMap)
        {
            // Ambient lighting
            pixelColor += evalEnvMapDir(envMap, r.dir);
        }
    }

    pixelColor /= (float)SAMPLES;
    atomic_add(&stats->samples, SAMPLES);
    return pixelColor;
}

// Path tracing!
inline float3 tracePath(float2 pos, uint iter, global uchar *texData, global TexDescriptor *textures, global PointLight *lights, global Triangle *tris, global Material *materials, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params, global RenderStats *stats)
{
    uint seed = get_global_id(1) * params->width + get_global_id(0) + iter * params->width * params->height; // unique for each pixel

    // Jittered AA
    pos += (float2)(rand(&seed), rand(&seed));

    Ray r = getCameraRay(pos, params);
    Hit hit = raycast(&r, FLT_MAX, tris, nodes, indices, params, params->sampleImpl);
    atomic_inc(&stats->primaryRays);

    if (hit.i < 0) return evalEnvMapDir(envMap, r.dir);

    AreaLight areaLight = params->areaLight;
    float3 emission = areaLight.E;
    float3 nLight = areaLight.N;

    // Path state
    float3 Ei = (float3)(0.0f);         // Irradiance
    float3 throughput = (float3)(1.0f); // BRDF
    float prob = 1.0f;                  // PDF
    float lastPdfW = 1.0f;              // for MIS
    bool lastSpecular = false;          // prevents NEE
    int i = 0;

    const int MAX_BOUNCES = params->maxBounces;
    while(i < MAX_BOUNCES && prob > 0.0f)
    {
        float3 Kd, Ks, n; // fetched from texture/material
        float refr = 0.0f;
        if (hit.areaLightHit)
            n = hit.N;
        else
            getMaterialParameters(hit, tris, materials, texData, textures, &Kd, &n, &Ks, &refr);

        // Backside of triangle hit
        bool backside = (dot(n, r.dir) > 0.0f);
        if (backside && !hit.areaLightHit)
        {
            n *= -1;
        }


        // Implicit area light sampling (light hit by chance)
        if (hit.areaLightHit)
        {
            float misWeight = 1.0f;
            if (params->sampleExpl && i > 0 && !lastSpecular) // not very direct + MIS needed
            {
                const float directPdfA = 1.0f / (4.0f * params->areaLight.size.x * params->areaLight.size.y);
                const float directPdfW = pdfAtoW(directPdfA, length(hit.P - r.orig), dot(normalize(-r.dir), hit.N));
                const float lightPickProb = 1.0f;
                misWeight = lastPdfW / (lastPdfW + directPdfW * lightPickProb);
            }

            // pdf (i.e. extension ray pdf = lastPdfW) included in prob
            Ei += throughput * misWeight * emission / prob;
            break; // lights don't reflect
        }


        /* REFRACTION */
        if (refr > 1.0f && i <= MAX_BOUNCES)
        {
            float3 orig;// , dir;
            const float EPS_REFR = 1e-5f;
            float cosI = dot(-normalize(r.dir), n);

			float n1 = 1.0f, n2 = refr;
			if (backside) swap_m(n1, n2, float); // inside of material

            float cosT = 1.0f - pow(n1 / n2, 2.0f) * (1.0f - pow(cosI, 2.0f));
            float raylen = length(r.dir);

            // Total internal reflection
            if (cosT < 0.0f)
            {
                orig = hit.P + EPS_REFR * n;
                r.dir = raylen * reflect(normalize(r.dir), n);
            }
            else
            {
                cosT = sqrt(cosT);
                // Fresnel: reflectance for unpolarized light
                float fr = 0.5f * (pow((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT), 2.0f) + pow((n2 * cosI - n1 * cosT) / (n1 * cosT + n2 * cosI), 2.0f));

                if (rand(&seed) < fr)
                {
                    // Reflection
                    orig = hit.P + EPS_REFR * n;
                    r.dir = raylen * reflect(normalize(r.dir), n);
					// fr in pdf and T cancel out
                }
                else
                {
                    // Refraction
                    orig = hit.P - EPS_REFR * n;
                    r.dir = raylen * (normalize(r.dir) * (n1 / n2) + n * ((n1 / n2) * cosI - cosT));
					// (1 - fr) in pdf and T cancel out
                }
            }

            // Cast continuation ray
            r.orig = orig;
            hit = raycast(&r, FLT_MAX, tris, nodes, indices, params, params->sampleImpl);
            atomic_inc(&stats->extensionRays);
            if (hit.i < 0) break;

            lastSpecular = true;
            throughput *= Ks; // simulate absorption
            i++;
            continue;
        }

        lastSpecular = false;
        float3 orig = hit.P - 1e-3f * r.dir;  // avoid self-shadowing

        // Explicit light source sampling (next event estimation)
        if (params->sampleExpl && !lastSpecular)
        {
            // Sample light source
            float directPdfA;
            float3 posL;
            sampleAreaLight(areaLight, &directPdfA, &posL, &seed);

            // Shadow ray
            float3 L = posL - orig;
            float lenL = length(L);
            L = normalize(L);
            Ray rLight = { orig, L };
            bool occluded = bvh_occluded(&rLight, &lenL, tris, nodes, indices); // no need to check area light
            atomic_inc(&stats->shadowRays);

            float cosLight = max(dot(nLight, -L), 0.0f);

            if (!occluded && cosLight > 1e-6f) // front of light accessible
            {
                const float lightPickProb = 1.0f;
                const float3 brdf = Kd / M_PI_F; // Kd = reflectivity/albedo
                float cosTh = max(0.0f, dot(L, n)); // cos at surface
                float directPdfW = pdfAtoW(directPdfA, lenL, cosLight); // 'how small area light looks'
                float bsdfPdfW = max(0.0f, cosTh / M_PI_F);
                
                float weight = 1.0f;
                if (params->sampleImpl)
                {
                    weight = (directPdfW * lightPickProb) / (directPdfW * lightPickProb + bsdfPdfW);
                }

                Ei += brdf * throughput * emission * weight * cosTh / (lightPickProb * directPdfW * prob);
            }
        }

        if (i+1 == MAX_BOUNCES) break;

        // Continuation ray
        float pdf, costh;
        sampleHemisphere(&(hit.P), &n, &costh, &seed, &pdf, &r.dir); // direction updated

        float3 brdf = Kd / M_PI_F;
        throughput *= brdf * costh;
        prob *= pdf;
        lastPdfW = pdf;

        r.orig = orig;
        hit = raycast(&r, FLT_MAX, tris, nodes, indices, params, params->sampleImpl);
        atomic_inc(&stats->extensionRays);

        if (hit.i < 0)
            break;

        i++;
    }

    atomic_inc(&stats->samples);
    return Ei;
}

/* 
OPENCL MEMORY SPACES:
| OpenCL   | OpenCL keyword | Scope           | CUDA     | CUDA keyword |
|----------+----------------+-----------------+----------+--------------+
| Global   | __global       | Kernel-wide     | Global   |              |
| Constant | __constant     | Kernel-wide     | Constant | __constant__ |
| Local    | __local        | Work-group-wide | Shared   | __shared__   |
| Private  | __private      | Work-item-wide  | Local    |              |
*/
kernel void trace(read_only image2d_t src, write_only image2d_t dst, global uchar *texData, global TexDescriptor *textures, global PointLight *lights, global Triangle *tris, global Material *materials, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params, global RenderStats *stats, uint iteration)
{
    const uint x = get_global_id(0); // left to right
    const uint y = get_global_id(1); // bottom to top

    if(x >= params->width || y >= params->height) return;

    // Ray tracing
    //float3 pixelColor = traceRay((float2)(x, y), texData, textures, lights, tris, materials, nodes, indices, envMap, params, stats);

    // Path tracing + accumulation
    //*
    float3 pixelColor = tracePath((float2)(x, y), iteration, texData, textures, lights, tris, materials, nodes, indices, envMap, params, stats);
    const float tex_weight = iteration * native_recip((float)(iteration) + 1.0f);
    float3 prev = read_imagef(src, (int2)(x, y)).xyz;
    pixelColor = mix(pixelColor, prev, tex_weight); // don't clamp => can be tonemapped e.g. in GL shader
    //*/

    write_imagef(dst, (int2)(x, y), (float4)(pixelColor, 1.0f));
}
