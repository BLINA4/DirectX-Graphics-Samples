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
        0.05f,
        direction,
        FLT_MAX };
    HitInfo payload;

    // Initialize random numbers generator
    RngStateType rngState = initRNG(DispatchRaysIndex().xy, DispatchRaysDimensions().xy, g_dynamic.frameNumber);

    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < g_dynamic.recursionDepth; bounce++)
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
        Light sun = g_dynamic.lights[0];
        Light pl = g_dynamic.lights[1];

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

            //if (length(radiance) > length(float3(1.0f, 1.0f, 1.0f)))
            //    g_screenOutput[DispatchRaysIndex().xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
            //else
            //    g_screenOutput[DispatchRaysIndex().xy] = float4(radiance, 1.0f);

            getLightData(pl, payload.hitPosition, lightVector, lightDistance);
            L = normalize(lightVector);

            // Cast shadow ray towards the selected light
            if (castShadowRay(payload.hitPosition, geometryNormal, L, lightDistance))
            {
                // If light is not in shadow, evaluate BRDF and accumulate its contribution into radiance
                radiance += throughput * evalCombinedBRDF(shadingNormal, L, V, albedo, metallness, roughness)
                    * (getLightIntensityAtPoint(pl, lightDistance) * lightWeight);
            }
        }

        // Terminate loop early on last bounce (we don't need to sample BRDF)
        if (bounce == g_dynamic.recursionDepth - 1) 
            break;

        // Russian Roulette
        //if (bounce > 2) 
        //{
        //    float rrProbability = min(0.95f, luminance(throughput));
        //    if (rrProbability < rand(rngState)) break;
        //    else throughput /= rrProbability;
        //}

        // Sample BRDF to generate the next ray
        // First, figure out whether to sample diffuse or specular BRDF
        int brdfType;
        if (metallness == 1.0f && roughness == 0.0f)
        {
            // Fast path for mirrors
            brdfType = SPECULAR_TYPE;
        }
        else 
        {
            // Decide whether to sample diffuse or specular BRDF (based on Fresnel term)
            float brdfProbability = getBrdfProbability(albedo, metallness, roughness, V, shadingNormal);

            if (rand(rngState) < brdfProbability) 
            {
                brdfType = SPECULAR_TYPE;
                throughput /= brdfProbability;
            }
            else 
            {
                brdfType = DIFFUSE_TYPE;
                throughput /= (1.0f - brdfProbability);
            }
        }

        // Run importance sampling of selected BRDF to generate reflecting ray direction
        float3 brdfWeight;
        float2 u = float2(rand(rngState), rand(rngState));
        if (!evalIndirectCombinedBRDF(u, shadingNormal, geometryNormal, V, albedo,
                                      metallness, roughness, brdfType, rayDesc.Direction, brdfWeight))
            break; // Ray was eaten by the surface :(

        // Account for surface properties using the BRDF "weight"
        throughput *= brdfWeight;

        rayDesc.Origin = offsetRay(payload.hitPosition, geometryNormal);
    }

    //if (length(throughput) > length(float3(1.0f, 1.0f, 1.0f)))
    //    g_screenOutput[DispatchRaysIndex().xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);

    // Temporal accumulation
    float4 prev = g_screenAccum[DispatchRaysIndex().xy] * (g_dynamic.accumulateNumber - 1);
    float4 accumulatedColor = (prev + float4(radiance, 1.0f)) / g_dynamic.accumulateNumber;
    g_screenAccum[DispatchRaysIndex().xy] = accumulatedColor;

    g_screenOutput[DispatchRaysIndex().xy] = float4(accumulatedColor.xyz, 1.0f);
}