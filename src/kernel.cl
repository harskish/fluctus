kernel void trace(write_only image2d_t img) {
    int i = get_global_id(0);
    int2 dim = get_image_dim(img);
    if(i < dim.x * dim.y) {
        int2 pos = (int2)(i % dim.x, i / dim.x);

        float color = (float)i / (dim.x * dim.y);
        if(dim.x * dim.y <= 100) {
            printf("Position: [%d, %d], color: %.3f\n", pos.x, pos.y, color);
        }

        float4 c = (float4)((float)i / (dim.x * dim.y), 0.0f, 0.0f, 1.0f);
        write_imagef(img, pos, c);
    }
}