#define HLSL
#define PATHTRACE

#include "RayTracingHlslCompat.h"
#include "ModelViewerRaytracing.h"

[shader("miss")]
void Miss( inout HitInfo payload )
{
    payload.materialID = INVALID_ID;
}