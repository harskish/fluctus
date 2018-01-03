#ifndef CL_BXDF_TYPES
#define CL_BXDF_TYPES

#define BXDF_DIFFUSE                (1 << 1)
#define BXDF_GLOSSY                 (1 << 2)
#define BXDF_GGX_ROUGH_REFLECTION   (1 << 3)
#define BXDF_IDEAL_REFLECTION       (1 << 4)
#define BXDF_GGX_ROUGH_DIELECTRIC   (1 << 5)
#define BXDF_IDEAL_DIELECTRIC       (1 << 6)
#define BXDF_EMISSIVE               (1 << 7)
#define BXDF_IS_SINGULAR(t) ((t & (BXDF_IDEAL_REFLECTION | BXDF_IDEAL_DIELECTRIC)) != 0)

#endif