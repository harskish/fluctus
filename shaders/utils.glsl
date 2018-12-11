struct Ray {
    vec3 origin;
    vec3 direction;
};

Ray generate_ray(vec2 film_position) {
    const float tan_fovy_over_2 = 0.414; // tan(45/2)

    vec2 uv = film_position / vec2(gl_LaunchSizeNVX.xy);
    float aspect_ratio = float(gl_LaunchSizeNVX.x) / float(gl_LaunchSizeNVX.y);
    float horz_half_dist = aspect_ratio * tan_fovy_over_2;
    float vert_half_dist = tan_fovy_over_2;
    vec2 uv2 = 2.0 * uv - 1.0;
    float dir_x = uv2.x * horz_half_dist;
    float dir_y = -uv2.y * vert_half_dist;

    Ray ray;
    ray.origin = vec3(0, 0, 3);
    ray.direction = normalize(vec3(dir_x, dir_y, -1.f));
    return ray;
}

float srgb_encode(float c) {
    if (c <= 0.0031308f)
        return 12.92f * c;
    else
        return 1.055f * pow(c, 1.f/2.4f) - 0.055f;
}

vec3 srgb_encode(vec3 c) {
    return vec3(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b));
}
