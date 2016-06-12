#ifdef GPU
typedef float4 cl_float4;
typedef float cl_float;
#endif

typedef struct
{
    cl_float4 orig;
    cl_float4 dir;
    // float4 inv_dir;
} Ray;

typedef struct
{
    cl_float R;     // 4B (padded to 16B?)
    cl_float4 pos;  // 16B
    cl_float4 Kd;   // 16B
} Sphere;           // 48B