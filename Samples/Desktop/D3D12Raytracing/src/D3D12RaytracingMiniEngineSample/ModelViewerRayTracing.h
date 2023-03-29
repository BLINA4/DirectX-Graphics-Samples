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

#ifndef PI

#define PI 3.141592653589f

#endif

// Ray miss identificator
#define INVALID_ID -1

// Indexes for ray types
#define STANDARD_RAY_INDEX 0
#define SHADOW_RAY_INDEX 1

// Light sources types
#define POINT_LIGHT 1
#define DIRECTIONAL_LIGHT 2

// Brdf reflection types
#define DIFFUSE_TYPE 1
#define SPECULAR_TYPE 2

#define RngStateType uint4

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
    float3  Albedo;
    float3  MetallRoughness;
    float3  hitPosition;
    uint    materialID;
    uint    triangleID;
    float2  uvs;
};

StructuredBuffer<RayTraceMeshInfo> g_meshInfo           : register(t1);
ByteAddressBuffer                  g_indices            : register(t2);
ByteAddressBuffer                  g_attributes         : register(t3);
Texture2D<float>                   texShadow            : register(t4);
Texture2D<float>                   texSSAO              : register(t5);
SamplerState                       g_s0                 : register(s0);
SamplerComparisonState             shadowSampler        : register(s1);

Texture2D<float4>                  g_localTexture       : register(t6);
Texture2D<float4>                  g_localMetalRough    : register(t7);
Texture2D<float4>                  g_localNormal        : register(t8);

Texture2D<float4>                  normals              : register(t13);

#endif

#endif

#pragma once

struct Light
{
    float3  position;
    float   pad0;
    uint    type;
    float3  pad1;
    float3  intensity;
    float   pad2;
};

// Volatile part (can be split into its own CBV). 
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3   worldCameraPosition;
    float    padding0;
    float2   resolution;
    float2   padding1;
    Light    lights[4];
    uint     recursionDepth;
    uint     frameNumber;
    uint     accumulateNumber;
    uint     lightsNumber;
};

#ifdef HLSL
#ifndef SINGLE
static const float FLT_MAX = asfloat(0x7F7FFFFF);
#endif

RaytracingAccelerationStructure g_accel : register(t0);

RWTexture2D<float4> g_screenOutput : register(u2);
RWTexture2D<float4> g_screenAccum  : register(u3);

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

#ifdef PATHTRACE

inline uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v = v ^ (v >> 16u);

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

// 32-bit Xorshift random number generator
inline uint xorshift(inout uint rngState)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}

// Jenkins's "one at a time" hash function
inline uint jenkinsHash(uint x) 
{
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
inline float uintToFloat(uint x) 
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG for given pixel, and frame number (PCG version)
inline RngStateType initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber) 
{
    return RngStateType(pixelCoords.xy, frameNumber, 0); //< Seed for PCG uses a sequential sample number in 4th channel, which increments on every RNG call and starts from 0
}

// Return random float in <0; 1) range  (PCG version)
inline float rand(inout RngStateType rngState) 
{
    rngState.w++; //< Increment sample index
    return uintToFloat(pcg4d(rngState).x);
}

inline uint3 Load3x16BitIndices(uint offsetBytes)
{
    const uint dwordAlignedOffset = offsetBytes & ~3;

    const uint2 four16BitIndices = g_indices.Load2(dwordAlignedOffset);

    uint3 indices;

    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}

inline float2 GetUVAttribute(uint byteOffset)
{
    return asfloat(g_attributes.Load2(byteOffset));
}

inline float3 RayPlaneIntersection(float3 planeOrigin, float3 planeNormal, float3 rayOrigin, float3 rayDirection)
{
    float t = dot(-planeNormal, rayOrigin - planeOrigin) / dot(planeNormal, rayDirection);
    return rayOrigin + rayDirection * t;
}

inline float3 BarycentricCoordinates(float3 pt, float3 v0, float3 v1, float3 v2)
{
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 e2 = pt - v0;
    float d00 = dot(e0, e0);
    float d01 = dot(e0, e1);
    float d11 = dot(e1, e1);
    float d20 = dot(e2, e0);
    float d21 = dot(e2, e1);
    float denom = 1.0 / (d00 * d11 - d01 * d01);
    float v = (d11 * d20 - d01 * d21) * denom;
    float w = (d00 * d21 - d01 * d20) * denom;
    float u = 1.0 - v - w;
    return float3(u, v, w);
}

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

inline float4 encodeNormals(float3 geometryNormal, float3 shadingNormal) 
{
    return float4(encodeNormalOctahedron(geometryNormal), encodeNormalOctahedron(shadingNormal));
}

inline void decodeNormals(float4 encodedNormals, out float3 geometryNormal, out float3 shadingNormal) 
{
    geometryNormal = decodeNormalOctahedron(encodedNormals.xy);
    shadingNormal = decodeNormalOctahedron(encodedNormals.zw);
}

inline void getLightData(Light light, float3 hitPosition, out float3 lightVector, out float lightDistance) 
{
    if (light.type == POINT_LIGHT) 
    {
        lightVector = light.position - hitPosition;
        lightDistance = length(lightVector);
    }
    else if (light.type == DIRECTIONAL_LIGHT) 
    {
        lightVector = light.position; //< We use position field to store direction for directional light
        lightDistance = FLT_MAX;
    }
    else 
    {
        lightDistance = FLT_MAX;
        lightVector = float3(0.0f, 1.0f, 0.0f);
    }
}

// Returns intensity of given light at specified distance
inline float3 getLightIntensityAtPoint(Light light, float distance) 
{
    if (light.type == POINT_LIGHT) 
    {
        // Cem Yuksel's improved attenuation avoiding singularity at zero distance
        const float radius = 0.5f; //< We hardcode radius at 0.5, but this should be a light parameter
        const float radiusSquared = radius * radius;
        const float distanceSquared = distance * distance;
        const float attenuation = 2.0f / (distanceSquared + radiusSquared + distance * sqrt(distanceSquared + radiusSquared));

        return light.intensity * attenuation;
    }
    else if (light.type == DIRECTIONAL_LIGHT) 
    {
        return light.intensity;
    }
    else 
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
}


inline float3 offsetRay(const float3 p, const float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

#include "brdf.h"

bool castShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float TMax)
{
    RayDesc ray;
    ray.Origin = offsetRay(hitPosition, surfaceNormal);
    ray.Direction = directionToLight;
    ray.TMin = 0.05f;
    ray.TMax = TMax;

    HitInfo payload;
    payload.materialID = INVALID_ID;

    // Trace the ray
    TraceRay(
        g_accel,
        RAY_FLAG_NONE,
        0xFF,
        SHADOW_RAY_INDEX,
        0,
        SHADOW_RAY_INDEX,
        ray,
        payload);

    return payload.materialID == INVALID_ID;
}

// Samples a random light from the pool of all lights using simplest uniform distirbution
bool sampleLightUniform(inout RngStateType rngState, float3 hitPosition, float3 surfaceNormal, out Light light, out float lightSampleWeight) {

    if (g_dynamic.lightsNumber == 0) return false;

    uint randomLightIndex = min(g_dynamic.lightsNumber - 1, uint(rand(rngState) * g_dynamic.lightsNumber));
    light = g_dynamic.lights[randomLightIndex];

    // PDF of uniform distribution is (1/light count). Reciprocal of that PDF (simply a light count) is a weight of this sample
    lightSampleWeight = float(g_dynamic.lightsNumber);

    return true;
}

// Samples a random light from the pool of all lights using RIS (Resampled Importance Sampling)
bool sampleLightRIS(inout RngStateType rngState, float3 hitPosition, float3 surfaceNormal, out Light selectedSample, out float lightSampleWeight) {

    if (g_dynamic.lightsNumber == 0) return false;

    selectedSample = (Light)0;
    float totalWeights = 0.0f;
    float samplePdfG = 0.0f;

    for (int i = 0; i < 8; i++) {

        float candidateWeight;
        Light candidate;
        if (sampleLightUniform(rngState, hitPosition, surfaceNormal, candidate, candidateWeight)) {

            float3	lightVector;
            float lightDistance;
            getLightData(candidate, hitPosition, lightVector, lightDistance);

            // Ignore backfacing light
            float3 L = normalize(lightVector);
            if (dot(surfaceNormal, L) < 0.00001f) continue;

            // Casting a shadow ray for all candidates here is expensive, but can significantly decrease noise
            if (!castShadowRay(hitPosition, surfaceNormal, L, lightDistance)) continue;

            float candidatePdfG = luminance(getLightIntensityAtPoint(candidate, length(lightVector)));
            const float candidateRISWeight = candidatePdfG * candidateWeight;

            totalWeights += candidateRISWeight;
            if (rand(rngState) < (candidateRISWeight / totalWeights)) {
                selectedSample = candidate;
                samplePdfG = candidatePdfG;
            }
        }
    }

    if (totalWeights == 0.0f) {
        return false;
    }
    else {
        lightSampleWeight = (totalWeights / float(8)) / samplePdfG;
        return true;
    }
}

#endif

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
