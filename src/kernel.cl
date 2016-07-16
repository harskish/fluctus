#include "geom.h"

#define printVec(title, v) printf("%s: { %.4f, %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z, (v).w)
#define dbg(expr) if(get_global_id(0) == 0 && get_global_id(1) == 0) { expr; }
//#define dbg(expr) if(false) { expr; }

inline void swap(float *a, float *b)
{
  float tmp = *b;
  *b = *a;
  *a = tmp;
}

inline bool sphereIntersect(Ray *r, constant Sphere *s, float *t)
{
    float t0, t1;
    float radius2 = s->R * s->R;

    // Geometric solution
    float3 L = s->P - r->orig;
    float tca = dot(L, r->dir);
    float d2 = dot(L, L) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    if (t0 > t1)
    {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    if (t0 < 0)
    {
        t0 = t1;
        if (t0 < 0) return false;
    }

    *t = t0;

    return true;
}

#define NORMAL_X ((float3)(-1, 0, 0))
#define NORMAL_Y ((float3)(0, -1, 0))
#define NORMAL_Z ((float3)(0, 0, -1))

// Assign normal according to face hit
inline bool intersectSlab(Ray *r, AABB *box, float *tminRet, float *tMaxRet, float3 *N)
{
	float3 n;
	float3 dinv = 1.0f / r->dir;

    // X-axis
	n = NORMAL_X;
	float dinvx = dinv.x;
	float tmin = (box->min.x - r->orig.x) * dinvx;
	float tmax = (box->max.x - r->orig.x) * dinvx;

	if (dinvx < 0)
	{
		swap(&tmin, &tmax);
		n *= -1.0f;
	}

	if (tmax < 0)
	{
		return false;
	}

	*N = n;

    // Y-axis
	n = NORMAL_Y;
	float dinvy = dinv.y;
	float tminy = (box->min.y - r->orig.y) * dinvy;
	float tmaxy = (box->max.y - r->orig.y) * dinvy;

	if (dinvy < 0)
	{
		swap(&tminy, &tmaxy);
		n *= -1.0f;
	}

	if (tmin > tmaxy || tmax < tminy)
	{
		return false;
	}

	if (tminy > tmin)
	{
		tmin = tminy;
		*N = n;
	}

	if (tmaxy < tmax)
	{
		tmax = tmaxy;
	}

	if (tmax < 0)
	{
		return false;
	}

    // Z-axis
	n = NORMAL_Z;
	float dinvz = dinv.z;
	float tminz = (box->min.z - r->orig.z) * dinvz;
	float tmaxz = (box->max.z - r->orig.z) * dinvz;

	if (dinvz < 0)
	{
		swap(&tminz, &tmaxz);
		n *= -1.0f;
	}

	if (tmin > tmaxz || tmax < tminz)
	{
		return false;
	}

	if (tminz > tmin)
	{
		tmin = tminz;
		*N = n;
	}

	if (tmaxz < tmax)
	{
		tmax = tmaxz;
	}

	if (tmax < 0)
	{
		return false;
	}

	// Assign output variables
	*tminRet = tmin;
	*tMaxRet = tmax;
	
	return true;
}

// Möller-Trumbore
inline bool intersectTriangle(Ray *r, Triangle *tri, float *tret, float *uret, float *vret)
{
	float3 s1 = tri->v1.p - tri->v0.p;;
	float3 s2 = tri->v2.p - tri->v0.p;
	float3 pvec = cross(r->dir, s2); // order matters!
	float det = dot(s1, pvec);

	// miss if det close to 0
	if (fabs(det) < FLT_EPSILON) return false;
	float iDet = 1.0f / det;

	float3 tvec = r->orig - tri->v0.p;
	float u = dot(tvec, pvec) * iDet;
	if (u < 0.0f || u > 1.0f) return false;

	float3 qvec = cross(tvec, s1); // order matters!
	float v = dot(r->dir, qvec) * iDet;
	if (v < 0.0f || u + v > 1.0f) return false;

	//float t = s2.dot(qvec) * iDet;
	float t = dot(s2, qvec) * iDet;

	if(t < 0.0f) return false;

	*tret = t;
	*uret = u;
	*vret = v;

	return true;
}

inline Ray getCameraRay(const uint x, const uint y, constant RenderParams *params)
{
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction

    // NDC-space, [0,1]x[0,1]
    float NDCx = (x + 0.5f) / params->width;
    float NDCy = (y + 0.5f) / params->height;

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

inline void calcNormalSphere(constant Sphere *scene, Hit *hit)
{
    hit->N = normalize(hit->P - (scene +hit->i)->P);
}

// Will be replaced with a BVH in the future...
// The ray length encodes the maximum intersection distance!
inline Hit raycast(Ray *r, float tMax, constant Sphere *scene, constant RenderParams *params)
{
    Hit hit = { (float3)(0.0f), (float3)(0.0f), tMax, -1 };

    // Spheres
    for(uint i = 0; i < params->n_objects; i++)
    {
        float t;
        bool found = sphereIntersect(r, &(scene[i]), &t);
        if(found && t < hit.t)
        {
            hit.t = t;
            hit.i = i;
            hit.P = r->orig + hit.t * r->dir;
            calcNormalSphere(scene, &hit);
        }
    }

    // AABBs
    AABB boxes[] = { {(float3)(-3, 1, -3), (float3)(-2, 2, -2)} };
    const uint n_boxes = 1;
    for(uint i = 0; i < n_boxes; i++)
    {
      float tmin, tmax;
      float3 N;
      bool found = intersectSlab(r, &(boxes[i]), &tmin, &tmax, &N); // fills in normal
      if(found && tmin < hit.t)
      {
          hit.t = tmin;
          hit.i = 2; // FOR TESTING ONLY!
          hit.P = r->orig + hit.t * r->dir;
          hit.N = N;
      }
    }

    // Triangles
    Vertex v0 = { (float3)(1.5f, 1.0f, -3.0f), (float3)(0.0f, 0.0f, 1.0f), (float3)(0.0f) };
    Vertex v1 = { (float3)(2.25f, 2.5f, -2.0f), (float3)(0.0f, 0.0f, 1.0f), (float3)(0.0f) };
    Vertex v2 = { (float3)(3.0f, 1.0f, -1.0f), (float3)(0.0f, 0.0f, 1.0f), (float3)(0.0f) };
    Triangle tris[] = { {v0, v1, v2} };
    const uint n_tris = 1;
    for(uint i = 0; i < n_tris; i++)
    {
      float t, u, v;
      float3 N;
      bool found = intersectTriangle(r, &(tris[i]), &t, &u, &v);
      if(found && t < hit.t)
      {
          hit.t = t;
          hit.i = i; // use Kd of sphere with same index
          hit.P = r->orig + hit.t * r->dir;
          hit.N = tris[i].v0.n; // interpolate!
      }
    }

    return hit;
}

inline float3 whittedShading(Hit *hit, constant Sphere *scene, constant Light *lights, constant RenderParams *params)
{
    float3 res = (float3)(0.0f);
    float3 lifted = hit->P + 1e-3f * hit->N;
    float3 V = normalize(params->camera.pos - hit->P);

    // Point light assumed for now
    for(uint i = 0; i < params->n_lights; i++)
    {
        float3 L = lights[i].pos - hit->P;
        float dist = length(L);
        L = normalize(L);

        Ray shadowRay = { lifted, L };
        Hit shdw = raycast(&shadowRay, dist, scene, params);
        float visibility = (shdw.i == -1) ? 1.0f : 0.0f; // early exits useless on GPU

        // Blinn-Phong

        // Testing material:
        float3 Ks = (float3)(1.0f);
        float glossiness = 0.025f; // probably not the right name...

        float3 H = normalize(L + V);
        float3 diffuse = scene[hit->i].Kd * max(0.0f, dot(L, hit->N));
        float3 specular = Ks * pow(max(0.0f, dot(hit->N, H)), 1.0f / glossiness);

        if(dot(hit->N, L) < 0) specular = (float3)(0.0f);

        float falloff = 1.0f / (dist * dist + 1e-5f);
        res += visibility * lights[i].intensity * falloff * (diffuse + specular);
    }

    return res;
}

kernel void trace(global float *out, constant Sphere *scene, constant Light *lights, constant RenderParams *params)
{
    const uint x = get_global_id(0); // left to right
    const uint y = get_global_id(1); // bottom to top

    if(x >= params->width || y >= params->height) return;

    Ray r = getCameraRay(x, y, params);
    Hit hit = raycast(&r, FLT_MAX, scene, params);

    float3 pixelColor = (hit.i == -1) ? (float3)(0.0f) : whittedShading(&hit, scene, lights, params);
    //float3 pixelColor = (hit.i != -1) ? scene[hit.i].Kd : (float3)(0.0f);

    //float3 prev = vload4((y * width + x), out);
    //float3 newCol = 0.005f * pixelColor + prev;

    vstore4((float4)(pixelColor, 0.0f), (y * params->width + x), out); // (value, offset, ptr)
}
