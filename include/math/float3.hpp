/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#ifndef FLOAT3_H
#define FLOAT3_H

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include <cmath>
#include <algorithm>

namespace FireRays
{
    class alignas(16) float3
    {
    public:
        float x, y, z, w;

        float3(float v = 0.f) : x(v), y(v), z(v), w(v) {}
        float3(float xx, float yy, float zz, float ww = 0.f) : x(xx), y(yy), z(zz), w(ww) {}

        float& operator [](int i)       { return *(&x + i); }
        float  operator [](int i) const { return *(&x + i); }
        float3 operator-() const        { return float3(-x, -y, -z); }

        float  sqnorm() const           { return x*x + y*y + z*z; }
        void   normalize()              { (*this)/=(std::sqrt(sqnorm()));}

        float3& operator += (float3 const& o) { x+=o.x; y+=o.y; z+= o.z; return *this;}
        float3& operator -= (float3 const& o) { x-=o.x; y-=o.y; z-= o.z; return *this;}
        float3& operator *= (float3 const& o) { x*=o.x; y*=o.y; z*= o.z; return *this;}
        float3& operator *= (float c) { x*=c; y*=c; z*= c; return *this;}
        float3& operator /= (float c) { float cinv = 1.f/c; x*=cinv; y*=cinv; z*=cinv; return *this;}

        void rotX(float phi);
        void rotY(float phi);
        void rotZ(float phi);
    };

    typedef float3 float4;


    inline float3 operator+(float3 const& v1, float3 const& v2)
    {
        float3 res = v1;
        return res+=v2;
    }

    inline float3 operator-(float3 const& v1, float3 const& v2)
    {
        float3 res = v1;
        return res-=v2;
    }

    inline float3 operator*(float3 const& v1, float3 const& v2)
    {
        float3 res = v1;
        return res*=v2;
    }

    inline float3 operator*(float3 const& v1, float c)
    {
        float3 res = v1;
        return res*=c;
    }

    inline float3 operator*(float c, float3 const& v1)
    {
        return operator*(v1, c);
    }

    inline float3 operator/(float3 const& v1, float c)
    {
        float3 res = v1;
        return res/=c;
    }

    inline float3 operator/(float c, float3 const& v1)
    {
        return float3(c/v1.x, c/v1.y, c/v1.z, c/v1.w);
    }

    inline float dot(float3 const& v1, float3 const& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    inline float3 normalize(float3 const& v)
    {
        float3 res = v;
        res.normalize();
        return res;
    }

    inline float3 cross(float3 const& v1, float3 const& v2)
    {
        // |i     j     k|
        // |v1x  v1y  v1z|
        // |v2x  v2y  v2z|
        return float3(v1.y * v2.z - v2.y * v1.z, v2.x * v1.z - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    }

    inline float3 vmin(float3 const& v1, float3 const& v2)
    {
        return float3(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
    }

    inline void vmin(float3 const& v1, float3 const& v2, float3& v)
    {
        v.x = std::min(v1.x, v2.x);
        v.y = std::min(v1.y, v2.y);
        v.z = std::min(v1.z, v2.z);
    }

    inline float3 vmax(float3 const& v1, float3 const& v2)
    {
        return float3(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
    }

    inline void vmax(float3 const& v1, float3 const& v2, float3& v)
    {
        v.x = std::max(v1.x, v2.x);
        v.y = std::max(v1.y, v2.y);
        v.z = std::max(v1.z, v2.z);
    }

    inline void float3::rotX(float phi)
    {
        phi *= (2 * M_PI) / 360;
        float tmpY = this->y;
        float tmpZ = this->z;
        this->y = cos(phi) * tmpY - sin(phi) * tmpZ;
        this->z = sin(phi) * tmpY + cos(phi) * tmpZ;
    }

    inline void float3::rotY(float phi)
    {
        phi *= (2 * M_PI) / 360;
        float tmpX = this->x;
        float tmpZ = this->z;
        this->z = -sin(phi) * tmpX + cos(phi) * tmpZ;
        this->x = cos(phi) * tmpX + sin(phi) * tmpZ;
    }

    inline void float3::rotZ(float phi)
    {
        phi *= (2 * M_PI) / 360;
        float tmpX = this->x;
        float tmpY = this->y;
        this->x = cos(phi) * tmpX - sin(phi) * tmpY;
        this->y = sin(phi) * tmpX + cos(phi) * tmpY;
    }

    inline float length(float3 const &v)
    {
        return sqrt(v.sqnorm());
    }
}

#endif // FLOAT3_H
