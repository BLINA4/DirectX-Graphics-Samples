#define HLSL
#define PATHTRACE

#include "ModelViewerRayTracing.h"
#include "RayTracingHlslCompat.h"

cbuffer Material                                  : register(b3)
{
    uint MaterialID;
}

StructuredBuffer<RayTraceMeshInfo> g_meshInfo     : register(t1);
ByteAddressBuffer                  g_indices      : register(t2);
ByteAddressBuffer                  g_attributes   : register(t3);
Texture2D<float>                   texShadow      : register(t4);
Texture2D<float>                   texSSAO        : register(t5);
SamplerState                       g_s0           : register(s0);
SamplerComparisonState             shadowSampler  : register(s1);

Texture2D<float4>                  g_localTexture : register(t6);
Texture2D<float4>                  g_localNormal  : register(t7);

Texture2D<float4>                  normals        : register(t13);

[shader("raygeneration")]
void RayGen()
{
    float3 origin, direction;
    GenerateCameraRay(DispatchRaysIndex().xy, origin, direction);

    RayDesc rayDesc = { origin,
        0.0f,
        direction,
        FLT_MAX };
    HitInfo payload;
    TraceRay(g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);

    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < 1; bounce++)
    {
        TraceRay(g_accel, RAY_FLAG_NONE, 0xFF, STANDARD_RAY_INDEX, 1, STANDARD_RAY_INDEX, rayDesc, payload);

        // On a miss, load the sky value and break out of the ray tracing loop
        if (payload.materialID == INVALID_ID)
        {
            radiance += throughput * /*loadSkyValue(ray.Direction)*/ float3(0.3f, 0.5f, 0.7f);
            break;
        }

        // Decode normals and flip them towards the incident ray direction (needed for backfacing triangles)
        float3 geometryNormal;
        float3 shadingNormal;
        decodeNormals(payload.encodedNormals, geometryNormal, shadingNormal);

        float3 V = -rayDesc.Direction;
        if (dot(geometryNormal, V) < 0.0f) geometryNormal = -geometryNormal;
        if (dot(geometryNormal, shadingNormal) < 0.0f) shadingNormal = -shadingNormal;

        radiance = throughput * shadingNormal;

        // Load material values

        // Account for emissive surfaces
        //radiance += throughput * material.emissive;

        // Evaluate direct light (next event estimation), start by sampling one light 
        //Light light;
        //float lightWeight;
        //if (sampleLightRIS(rngState, payload.hitPosition, geometryNormal, light, lightWeight))
        //{
            // Prepare data needed to evaluate the light
        //    float3 lightVector;
        //    float lightDistance;
        //    getLightData(light, payload.hitPosition, lightVector, lightDistance);
        //    float3 L = normalize(lightVector);

            // Cast shadow ray towards the selected light
        //    if (SHADOW_RAY_IN_RIS || castShadowRay(payload.hitPosition, geometryNormal, L, lightDistance))
        //    {
                // If light is not in shadow, evaluate BRDF and accumulate its contribution into radiance
        //        radiance += throughput * evalCombinedBRDF(shadingNormal, L, V, material) * (getLightIntensityAtPoint(light, lightDistance) * lightWeight);
        //    }
        //}
        
        g_screenOutput[DispatchRaysIndex().xy] = float4(radiance, 1.0f);
    }
}