#ifndef HLSL
#include "HlslCompat.h"
#endif

#if __cplusplus
#pragma once
#define matrix DirectX::XMMATRIX
#define float4 DirectX::XMFLOAT4
#define float3 DirectX::XMFLOAT3
#define float2 DirectX::XMFLOAT2
#define float16_t uint16_t
#define float16_t2 uint32_t
#define uint uint32_t
#define OUT_PARAMETER(X) X&
#else
#define OUT_PARAMETER(X) out X
#endif

#define INVALID_ID -1
#define STANDARD_RAY_INDEX 0
#define SHADOW_RAY_INDEX 1

#ifdef HLSL

#ifndef PATHTRACE

struct RayPayload
{
    bool SkipShading;
    float RayHitT;
};

#else

struct HitInfo
{
    float4  encodedNormals;
    float3  hitPosition;
    uint    materialID;
    float2  uvs;
};

#endif

#endif

#pragma once
// Volatile part (can be split into its own CBV). 
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3   worldCameraPosition;
    uint     padding;
    float2   resolution;
};
#ifdef HLSL
#ifndef SINGLE
static const float FLT_MAX = asfloat(0x7F7FFFFF);
#endif

RaytracingAccelerationStructure g_accel : register(t0);

RWTexture2D<float4> g_screenOutput : register(u2);

cbuffer HitShaderConstants : register(b0)
{
    float3 SunDirection;
    float3 SunColor;
    float3 AmbientColor;
    float4 ShadowTexelSize;
    float4x4 ModelToShadow;
    uint IsReflection;
    uint UseShadowRays;
}

cbuffer b1 : register(b1)
{
    DynamicCB g_dynamic;
};

// Helpers for octahedron encoding of normals
inline float2 octWrap(float2 v)
{
    return float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

inline float2 encodeNormalOctahedron(float3 n)
{
    float2 p = float2(n.x, n.y) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.0f) ? octWrap(p) : p;
    return p;
}

inline float3 decodeNormalOctahedron(float2 p)
{
    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float2 tmp = (n.z < 0.0f) ? octWrap(float2(n.x, n.y)) : float2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

inline float4 encodeNormals(float3 geometryNormal, float3 shadingNormal) {
    return float4(encodeNormalOctahedron(geometryNormal), encodeNormalOctahedron(shadingNormal));
}

inline void decodeNormals(float4 encodedNormals, out float3 geometryNormal, out float3 shadingNormal) {
    geometryNormal = decodeNormalOctahedron(encodedNormals.xy);
    shadingNormal = decodeNormalOctahedron(encodedNormals.zw);
}

inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5; // center in the middle of the pixel
    float2 screenPos = xy / g_dynamic.resolution * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    // Unproject into a ray
    float4 unprojected = mul(g_dynamic.cameraToWorld, float4(screenPos, 0, 1));
    float3 world = unprojected.xyz / unprojected.w;
    origin = g_dynamic.worldCameraPosition;
    direction = normalize(world - origin);
}
#endif
