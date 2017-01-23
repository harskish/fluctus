#ifndef CL_UTILS
#define CL_UTILS

#include "geom.h"
#include "random.cl"

#define printVec3(title, v) printf("%s: { %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z)
#define printVec4(title, v) printf("%s: { %.4f, %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z, (v).w)

#define swap_m(a, b, t) { t tmp = a; a = b; b = tmp; }

inline void swap(float *a, float *b)
{
  float tmp = *b;
  *b = *a;
  *a = tmp;
}

inline float3 lerp(float u, float v, float3 v1, float3 v2, float3 v3)
{
    return (1.0f - u - v) * v1 + u * v2 + v * v3;
}

inline float3 reflect(float3 dir, float3 n) // dir normalized?
{
    return dir - 2.0f * dot(dir, n) * n;
}

inline float3 refract(float3 dir, float3 n, float n1, float n2) // n1 = current, n2 = new
{
    float cosI = dot(-normalize(dir), n);
    float cosT = 1.0f - pow(n1 / n2, 2.0f) * (1.0f - pow(cosI, 2.0f));
	float raylen = length(dir);

	if (cosT < 0.0f)
		return raylen * reflect(normalize(dir), n); // Total internal reflection
	else
        return raylen * (normalize(dir) * (n1 / n2) + n * ((n1 / n2) * cosI - sqrt(cosT)));
}

inline void calcNormalSphere(global Sphere *scene, Hit *hit)
{
    hit->N = normalize(hit->P - (scene +hit->i)->P);
}

// Return a sample through the center of the Nth subpixel
inline float2 sampleRegular(int n, int dim)
{
    float d = 1.0f / dim;
	int x = n % dim;
	int y = n / dim;

    float xm1 = d * (x + 0.5f);
    float ym1 = d * (y + 0.5f);

    return (float2)(xm1, ym1);
}

inline void sampleHemisphere(float3 *pos, float3 *n, float *costh, uint *seed, float *p, float3 *dir)
{
    float r1 = 2.0f * M_PI_F * rand(seed);
    float r2 = rand(seed);
    float r2s = sqrt(r2);

    float3 w = *n;

    float3 u;
    if (fabs(w.x) > 0.1f) {
        float3 a = (float3)(0.0f, 1.0f, 0.0f);
        u = cross(a, w);
    }
    else {
        float3 a = (float3)(1.0f, 0.0f, 0.0f);
        u = cross(a, w);
    }
    u = normalize(u);

    float3 v = cross(w, u);
    
    u *= (cos(r1) * r2s);
    v *= (sin(r1) * r2s);
    w *= (sqrt(1 - r2));

    *dir = u + v + w;
    *costh = dot(*n, *dir);
    *p = *costh / M_PI_F; //pdf
}

inline int2 getTexelCoords(float2 uv, uint width, uint height)
{
    uv.x *= width;
    uv.y *= height;
    int tx = ((int)(floor(uv.x)) % width + width) % width;
    int ty = ((int)(floor(uv.y)) % height + height) % height;

    return clamp((int2)(tx + uv.x - floor(uv.x), ty + uv.y - floor(uv.y)), (int2)(0), (int2)(width - 1, height - 1));
    //return Vec2f(tx + uv.x - floor(uv.x), ty + uv.y - floor(uv.y)).clamp(Vec2f(0), Vec2f(size)-Vec2f(1));
}

inline float3 readTexture(float2 uvTex, TexDescriptor tex, global uchar *data)
{
    int2 coords = getTexelCoords(uvTex, tex.width, tex.height);
    global uchar *pix = data + tex.offset + coords.x * 4 + coords.y * tex.width * 4;
    float3 c = (float3)(*(pix + 0), *(pix + 1), *(pix + 2));

    return c / 255.0f;
}

inline void getMaterialParameters(Hit hit, global Triangle *tris, global Material *materials, global uchar *texData, global TexDescriptor *textures, float3 *Kd, float3 *N, float3 *Ks, float *refr)
{
    // Dummy method for now, should read from textures (if available)
    const Material mat = materials[hit.matId];

    *Kd = (mat.map_Kd != -1) ? readTexture(hit.uvTex, textures[mat.map_Kd], texData) : mat.Kd;
    //*Kd = mat.Kd;
    *Ks = mat.Ks;
    *refr = mat.Ni;
    *N = hit.N; // no normal maps yet
}

#endif