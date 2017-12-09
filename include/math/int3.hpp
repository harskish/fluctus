#ifndef INT3_H
#define INT3_H

#include <cmath>
#include <algorithm>

namespace FireRays
{
    class int3
    {
    public:
        int x, y, z, w;
        
        int3(int v = 0) : x(v), y(v), z(v), w(v) {}
        int3(int xx, int yy, int zz, int ww = 0) : x(xx), y(yy), z(zz), w(ww) {}
		int3(const float3 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

        int&  operator [](int i)       { return *(&x + i); }
        int   operator [](int i) const { return *(&x + i); }
        int3  operator-() const        { return int3(-x, -y, -z, -w); }

        int   sqnorm() const           { return x*x + y*y + z*z + w*w; }

        int3& operator += (int3 const& o) { x+=o.x; y+=o.y; z+=o.z; w+=o.w; return *this; }
        int3& operator -= (int3 const& o) { x-=o.x; y-=o.y; z-=o.z; w-=o.w; return *this; }
        int3& operator *= (int3 const& o) { x*=o.x; y*=o.y; z*=o.z; w*=o.w; return *this; }
        int3& operator *= (int c) { x*=c; y*=c; z*=c; w*=c; return *this; }
    };

    typedef int3 int4;

    inline int3 operator+(int3 const& v1, int3 const& v2)
    {
        int3 res = v1;
        return res+=v2;
    }

    inline int3 operator-(int3 const& v1, int3 const& v2)
    {
        int3 res = v1;
        return res-=v2;
    }

    inline int3 operator*(int3 const& v1, int3 const& v2)
    {
        int3 res = v1;
        return res*=v2;
    }

    inline int3 operator*(int3 const& v1, int c)
    {
        int3 res = v1;
        return res*=c;
    }

    inline int3 operator*(int c, int3 const& v1)
    {
        return operator*(v1, c);
    }

    inline int dot(int3 const& v1, int3 const& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
    }

    inline int3 vmin(int3 const& v1, int3 const& v2)
    {
        return int3(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z), std::min(v1.w, v2.w));
    }

    inline int3 vmax(int3 const& v1, int3 const& v2)
    {
        return int3(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z), std::max(v1.w, v2.w));
    }

	inline int3 vclamp(int3 const& v, int3 const& lo, int3 const& hi)
	{
		return vmax(lo, vmin(hi, v));
	}
}

#endif // INT3_H