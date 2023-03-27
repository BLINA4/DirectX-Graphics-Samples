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
    float2   padding2;
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

bool castShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float TMax)
{
    RayDesc ray;
    ray.Origin = offsetRay(hitPosition, surfaceNormal);
    ray.Direction = directionToLight;
    ray.TMin = 0.01f;
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

inline float4 getRotationToZAxis(float3 input)
{

    // Handle special case when input is exact or near opposite of (0, 0, 1)
    if (input.z < -0.99999f) return float4(1.0f, 0.0f, 0.0f, 0.0f);

    return normalize(float4(input.y, -input.x, 0.0f, 1.0f + input.z));
}

inline float4 getRotationFromZAxis(float3 input) 
{
    // Handle special case when input is exact or near opposite of (0, 0, 1)
    if (input.z < -0.99999f) return float4(1.0f, 0.0f, 0.0f, 0.0f);

    return normalize(float4(-input.y, input.x, 0.0f, 1.0f + input.z));
}

inline float4 invertRotation(float4 q)
{
    return float4(-q.x, -q.y, -q.z, q.w);
}

inline float3 rotatePoint(float4 q, float3 v) 
{
    const float3 qAxis = float3(q.x, q.y, q.z);
    return 2.0f * dot(qAxis, v) * qAxis + (q.w * q.w - dot(qAxis, qAxis)) * v + 2.0f * q.w * cross(qAxis, v);
}

inline float3 sampleHemisphere(float2 u, OUT_PARAMETER(float) pdf) 
{
    float a = sqrt(u.x);
    float b = 2 * PI * u.y;

    float3 result = float3(
        a * cos(b),
        a * sin(b),
        sqrt(1.0f - u.x));

    pdf = result.z / PI;

    return result;
}

inline float3 sampleHemisphere(float2 u) 
{
    float pdf;
    return sampleHemisphere(u, pdf);
}

// Data needed to evaluate BRDF (surface and material properties at given point + configuration of light and normal vectors)
struct BrdfData
{
    // Material properties
    float3 specularF0;
    float3 diffuseReflectance;

    // Roughnesses
    float roughness;    //< perceptively linear roughness (artist's input)
    float alpha;        //< linear roughness - often 'alpha' in specular BRDF equations
    float alphaSquared; //< alpha squared - pre-calculated value commonly used in BRDF equations

    // Commonly used terms for BRDF evaluation
    float3 F; //< Fresnel term

    // Vectors
    float3 V; //< Direction to viewer (or opposite direction of incident ray)
    float3 N; //< Shading normal
    float3 H; //< Half vector (microfacet normal)
    float3 L; //< Direction to light (or direction of reflecting ray)

    float NdotL;
    float NdotV;

    float LdotH;
    float NdotH;
    float VdotH;

    // True when V/L is backfacing wrt. shading normal N
    bool Vbackfacing;
    bool Lbackfacing;
};

inline float3 baseColorToSpecularF0(float3 baseColor, float metalness) 
{
    return lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metalness);
}

inline float3 baseColorToDiffuseReflectance(float3 baseColor, float metalness)
{
    return baseColor * (1.0f - metalness);
}

inline float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float shadowedF90(float3 F0) 
{
    const float t = (1.0f / 0.04f);
    return min(1.0f, t * luminance(F0));
}

inline float3 evalFresnelSchlick(float3 f0, float f90, float NdotS)
{
    return f0 + (f90 - f0) * pow(1.0f - NdotS, 5.0f);
}

inline float3 evalFresnel(float3 f0, float f90, float NdotS)
{
    // Default is Schlick's approximation
    return evalFresnelSchlick(f0, f90, NdotS);
}

inline float GGX_D(float alphaSquared, float NdotH) 
{
    float b = ((alphaSquared - 1.0f) * NdotH * NdotH + 1.0f);
    return alphaSquared / (PI * b * b);
}

inline float Smith_G1_GGX(float alpha, float NdotS, float alphaSquared, float NdotSSquared) 
{
    return 2.0f / (sqrt(((alphaSquared * (1.0f - NdotSSquared)) + NdotSSquared) / NdotSSquared) + 1.0f);
}

inline float Smith_G2_Over_G1_Height_Correlated(float alpha, float alphaSquared, float NdotL, float NdotV)
{
    float G1V = Smith_G1_GGX(alpha, NdotV, alphaSquared, NdotV * NdotV);
    float G1L = Smith_G1_GGX(alpha, NdotL, alphaSquared, NdotL * NdotL);
    return G1L / (G1V + G1L - G1V * G1L);
}

inline float Smith_G2_Height_Correlated_GGX_Lagarde(float alphaSquared, float NdotL, float NdotV)
{
    float a = NdotV * sqrt(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL));
    float b = NdotL * sqrt(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV));
    return 0.5f / (a + b);
}

// Evaluates microfacet specular BRDF
inline float3 evalMicrofacet(const BrdfData data) 
{
    float D = GGX_D(max(0.00001f, data.alphaSquared), data.NdotH);
    float G2 = Smith_G2_Height_Correlated_GGX_Lagarde(data.alphaSquared, data.NdotL, data.NdotV);

    return data.F * (G2 * D * data.NdotL);
}

inline float3 sampleGGXVNDF(float3 Ve, float2 alpha2D, float2 u)
{
    float3 Vh = normalize(float3(alpha2D.x * Ve.x, alpha2D.y * Ve.y, Ve.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    float r = sqrt(u.x);
    float phi = 2 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = lerp(sqrt(1.0f - t1 * t1), t2, s);

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return normalize(float3(alpha2D.x * Nh.x, alpha2D.y * Nh.y, max(0.0f, Nh.z)));
}

inline float3 sampleSpecularMicrofacet(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0,
                                       float2 u, OUT_PARAMETER(float3) weight)
{
    // Sample a microfacet normal (H) in local space
    float3 Hlocal;
    if (alpha == 0.0f)
    {
        Hlocal = float3(0.0f, 0.0f, 1.0f);
    }
    else
    {
        Hlocal = sampleGGXVNDF(Vlocal, float2(alpha, alpha), u);
    }

    // Reflect view direction to obtain light vector
    float3 Llocal = reflect(-Vlocal, Hlocal);

    // Note: HdotL is same as HdotV here
    // Clamp dot products here to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
    float HdotL = max(0.00001f, min(1.0f, dot(Hlocal, Llocal)));
    const float3 Nlocal = float3(0.0f, 0.0f, 1.0f);
    float NdotL = max(0.00001f, min(1.0f, dot(Nlocal, Llocal)));
    float NdotV = max(0.00001f, min(1.0f, dot(Nlocal, Vlocal)));
    float NdotH = max(0.00001f, min(1.0f, dot(Nlocal, Hlocal)));
    float3 F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL);

    // Calculate weight of the sample specific for selected sampling method 
    // (this is microfacet BRDF divided by PDF of sampling method - notice how most terms cancel out)
    weight = F * Smith_G2_Over_G1_Height_Correlated(alpha, alphaSquared, NdotL, NdotV);

    return Llocal;
}

inline BrdfData prepareBRDFData(float3 N, float3 L, float3 V, float3 baseColor, float metalness, float roughness) 
{
    BrdfData data;

    // Evaluate VNHL vectors
    data.V = V;
    data.N = N;
    data.H = normalize(L + V);
    data.L = L;

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    data.Vbackfacing = (NdotV <= 0.0f);
    data.Lbackfacing = (NdotL <= 0.0f);

    // Clamp NdotS to prevent numerical instability. Assume vectors below the hemisphere will be filtered using 'Vbackfacing' and 'Lbackfacing' flags
    data.NdotL = min(max(0.00001f, NdotL), 1.0f);
    data.NdotV = min(max(0.00001f, NdotV), 1.0f);

    data.LdotH = saturate(dot(L, data.H));
    data.NdotH = saturate(dot(N, data.H));
    data.VdotH = saturate(dot(V, data.H));

    // Unpack material properties
    data.specularF0 = baseColorToSpecularF0(baseColor, metalness);
    data.diffuseReflectance = baseColorToDiffuseReflectance(baseColor, metalness);

    // Unpack 'perceptively linear' -> 'linear' -> 'squared' roughness
    data.roughness = roughness;
    data.alpha = roughness * roughness;
    data.alphaSquared = data.alpha * data.alpha;

    // Pre-calculate some more BRDF terms
    data.F = evalFresnel(data.specularF0, shadowedF90(data.specularF0), data.LdotH);

    return data;
}

inline float3 evalLambertian(const BrdfData data)
{
    return data.diffuseReflectance * (1.0f / PI * data.NdotL);
}

// Calculates probability of selecting BRDF (specular or diffuse) using the approximate Fresnel term
inline float getBrdfProbability(float3 albedo, float metalness, float3 V, float3 shadingNormal)
{

    // Evaluate Fresnel term using the shading normal
    // Note: we use the shading normal instead of the microfacet normal (half-vector) for Fresnel term here. 
    // That's suboptimal for rough surfaces at grazing angles, but half-vector is yet unknown at this point
    float specularF0 = luminance(baseColorToSpecularF0(albedo, metalness));
    float diffuseReflectance = luminance(baseColorToDiffuseReflectance(albedo, metalness));
    float Fresnel = saturate(luminance(evalFresnel(specularF0, shadowedF90(specularF0), max(0.0f, dot(V, shadingNormal)))));

    // Approximate relative contribution of BRDFs using the Fresnel term
    float specular = Fresnel;
    float diffuse = diffuseReflectance * (1.0f - Fresnel); //< If diffuse term is weighted by Fresnel, apply it here as well

    // Return probability of selecting specular BRDF over diffuse BRDF
    float p = (specular / max(0.0001f, (specular + diffuse)));

    // Clamp probability to avoid undersampling of less prominent BRDF
    return clamp(p, 0.1f, 0.9f);
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for direct light)
inline float3 evalCombinedBRDF(float3 N, float3 L, float3 V, float3 albedo, float metalness, float roughness) 
{
    // Prepare data needed for BRDF evaluation - unpack material properties and evaluate commonly used terms (e.g. Fresnel, NdotL, ...)
    const BrdfData data = prepareBRDFData(N, L, V, albedo, metalness, roughness);

    // Ignore V and L rays "below" the hemisphere
    if (data.Vbackfacing || data.Lbackfacing) return float3(0.0f, 0.0f, 0.0f);

    // Eval specular and diffuse BRDFs
    float3 specular = evalMicrofacet(data);
    float3 diffuse = evalLambertian(data);

    // Specular is already multiplied by F, just attenuate diffuse
    return (float3(1.0f, 1.0f, 1.0f) - data.F) * diffuse + specular;

    //return diffuse + specular;
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for indirect light)
inline bool evalIndirectCombinedBRDF(float2 u, float3 shadingNormal, float3 geometryNormal, float3 V, 
                                     float3 albedo, float metalness, float roughness, const int brdfType,
                                     OUT_PARAMETER(float3) rayDirection, OUT_PARAMETER(float3) sampleWeight) 
{
    // Ignore incident ray coming from "below" the hemisphere
    if (dot(shadingNormal, V) <= 0.0f) return false;

    // Transform view direction into local space of our sampling routines 
    // (local space is oriented so that its positive Z axis points along the shading normal)
    float4 qRotationToZ = getRotationToZAxis(shadingNormal);
    float3 Vlocal = rotatePoint(qRotationToZ, V);
    const float3 Nlocal = float3(0.0f, 0.0f, 1.0f);

    float3 rayDirectionLocal = float3(0.0f, 0.0f, 0.0f);

    if (brdfType == DIFFUSE_TYPE) 
    {
        // Sample diffuse ray using cosine-weighted hemisphere sampling 
        rayDirectionLocal = sampleHemisphere(u);
        const BrdfData data = prepareBRDFData(Nlocal, rayDirectionLocal, Vlocal, albedo, metalness, roughness);

        // Function 'diffuseTerm' is predivided by PDF of sampling the cosine weighted hemisphere
        sampleWeight = data.diffuseReflectance * 1.0f;
	
        // Sample a half-vector of specular BRDF. Note that we're reusing random variable 'u' here, but correctly it should be an new independent random number
        float3 Hspecular = sampleGGXVNDF(Vlocal, float2(data.alpha, data.alpha), u);

        // Clamp HdotL to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
        float VdotH = max(0.00001f, min(1.0f, dot(Vlocal, Hspecular)));
        sampleWeight *= (float3(1.0f, 1.0f, 1.0f) - evalFresnel(data.specularF0, shadowedF90(data.specularF0), VdotH));
    }
    else if (brdfType == SPECULAR_TYPE) 
    {
        const BrdfData data = prepareBRDFData(Nlocal, float3(0.0f, 0.0f, 1.0f), Vlocal, albedo, metalness, roughness);
        rayDirectionLocal = sampleSpecularMicrofacet(Vlocal, data.alpha, data.alphaSquared, data.specularF0, u, sampleWeight);
    }

    // Prevent tracing direction with no contribution
    if (luminance(sampleWeight) == 0.0f) return false;

    // Transform sampled direction Llocal back to V vector space
    rayDirection = normalize(rotatePoint(invertRotation(qRotationToZ), rayDirectionLocal));

    // Prevent tracing direction "under" the hemisphere (behind the triangle)
    if (dot(geometryNormal, rayDirection) <= 0.0f) return false;

    return true;
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
