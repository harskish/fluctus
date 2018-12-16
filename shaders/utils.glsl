struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Hit
{
    vec3 P;
    vec3 N;
    vec2 uvTex;
    float t;
    int i; // index of hit triangle, -1 by default
    int areaLightHit;
    int matId; // index of hit material
};
#define EMPTY_HIT(tmax) { vec3(0.0), vec3(0.0), vec2(0.0), tmax, -1, 0, -1 }

struct PrimaryPayload {
    Hit hit;
    vec3 color;
};

float srgb_encode(float c) {
    if (c <= 0.0031308f)
        return 12.92f * c;
    else
        return 1.055f * pow(c, 1.f/2.4f) - 0.055f;
}

vec3 srgb_encode(vec3 c) {
    return vec3(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b));
}
