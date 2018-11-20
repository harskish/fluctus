#ifdef NVIDIA // NVIDIA only

#ifndef CL_PTX_INLINES
#define CL_PTX_INLINES

// Inline PTX assembly, exclusive to NVIDIA hardware.
// These could be replaced with OpenCL 2.0 work_group coop. functions,
// but NVIDIA probably won't ever support OCL2.

inline uint popcnt(const uint i)
{
    uint n;
    asm("popc.b32 %0, %1;" : "=r"(n) : "r"(i));
    return n;
}

inline uint activemask()
{
    uint n;
    asm("activemask.b32 %0;" : "=r"(n));
    return n;
}

inline uint ballot(const uint pred)
{
    uint ret;
    asm("{"
        ".reg .pred p1;\n\t"            // pred reg p1
        " setp.ne.u32 p1, %1, 0;\n\t"   // p1 = (pred != 0)
        " vote.ballot.b32 %0, p1;"      // ret = ballot(p1)
        "}"
        : "=r"(ret) : "r"(pred));
    return ret;
}

inline uint ballot_sync(const uint pred, const uint mask)
{
    uint ret;
    asm("{"
        " .reg .pred p1;\n\t"                 // pred reg p1
        " setp.ne.u32 p1, %1, 0;\n\t"         // p1 = (pred != 0)
        " vote.sync.ballot.b32 %0, p1, %2;"   // ret = ballot(p1)
        "}"
        : "=r"(ret) : "r"(pred), "r"(mask));
    return ret;
}

inline uint trailingZeros(uint x)
{
    return 31 - clz(x & -x);
}

inline uint leadingZeros(uint x)
{
    return clz(x);
}

// 1-based index of lsb set to 1
inline uint ffs(uint x)
{
    return trailingZeros(x) + 1;
}

inline uint laneid()
{
    return get_local_id(0) % 32;
}

inline uint shfl_idx_sync(uint membermask, uint a, uint b) // a = input, b = src lane
{
    const uint wanted_cval = 0x1f; // = 31 = 0b11111
    const uint wanted_segmask = 0x0;

    // Packed, mask[12:8] + clamp[4:0]
    const uint c = (wanted_cval | (wanted_segmask << 8)) & 0x1fff;

    /*
    uint cval = c & 0x1f; // [4:0]
    uint segmask = (c >> 8) & 0x1f; // [12:8]
    uint laneid = get_local_id(0) % 32;
    uint maxLane = (laneid & segmask) | (cval & ~segmask);
    uint minLane = (laneid & segmask);
    printf("c: 0x%x, cval: 0x%x, segmask: 0x%x, maxLane: %u, minLane: %u\n", c, cval, segmask, maxLane, minLane);
    */

    uint d;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;" : "=r"(d) : "r"(a), "r"(b), "r"(c), "r"(membermask));
    return d;
}

// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
inline uint atomicAggInc(global uint* ctr, const uint mask)
{
    uint leader = ffs(mask) - 1;
    uint laneid = get_local_id(0) % 32;
    uint res;

    //printf("Mask: 0x%x, leader: %u, laneid: %u, popcnt: %u, counter base: %u\n", mask, leader, laneid, popcnt(mask), *ctr);

    if (laneid == leader)
        res = atomic_add(ctr, popcnt(mask));

    res = shfl_idx_sync(mask, res, leader);

    uint targetIdx = res + popcnt(mask & ((1 << laneid) - 1));

    //if ((1 << laneid) & mask)
    //    printf("<%u> targetIdx: %u\n", laneid, targetIdx);

    return targetIdx;
}

#endif

#endif // NVIDIA only