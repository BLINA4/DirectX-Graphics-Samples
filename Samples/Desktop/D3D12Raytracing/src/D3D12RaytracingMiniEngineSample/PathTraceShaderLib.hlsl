#define HLSL
#define PATHTRACE

#include "RayTracingHlslCompat.h"
#include "ModelViewerRayTracing.h"

cbuffer Material : register(b3)
{
    uint MaterialID;
}

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
        if (dot(geometryNormal, V) < 0.0f)             geometryNormal = -geometryNormal;
        if (dot(geometryNormal, shadingNormal) < 0.0f) shadingNormal  = -shadingNormal;

        float3 albedo     = payload.Albedo;
        float  metallness = payload.MetallRoughness.b;
        float  roughness  = payload.MetallRoughness.g;

        // Account for emissive surfaces (currently not supported)
        // radiance += throughput * material.emissive;

        // Evaluate direct light (next event estimation), start by sampling one light 
        Light sun;
        sun.type = DIRECTIONAL_LIGHT;
        sun.intensity = float3(0.7f, 0.7f, 0.7f);
        sun.position = float3(300.0f, 1000.0f, 100.0f);
        sun.pad = 0;

        float lightWeight = 1.0f;
        if (true /*sampleLightRIS(rngState, payload.hitPosition, geometryNormal, light, lightWeight)*/)
        {
            // Prepare data needed to evaluate the light
            float3 lightVector;
            float lightDistance;
            getLightData(sun, payload.hitPosition, lightVector, lightDistance);
            float3 L = normalize(lightVector);

            // Cast shadow ray towards the selected light
            if (castShadowRay(payload.hitPosition, geometryNormal, L, lightDistance))
            {
                // If light is not in shadow, evaluate BRDF and accumulate its contribution into radiance
                radiance += throughput * evalCombinedBRDF(shadingNormal, L, V, albedo, metallness, roughness)
                    * (getLightIntensityAtPoint(sun, lightDistance) * lightWeight);
            }

            g_screenOutput[DispatchRaysIndex().xy] = float4(radiance, 1.0f);
        }
    }

    //g_screenOutput[DispatchRaysIndex().xy] = float4(radiance, 1.0f);
}