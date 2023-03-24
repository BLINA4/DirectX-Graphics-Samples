#define HLSL
#define PATHTRACE

#include "ModelViewerRaytracing.h"

[shader("miss")]
void Miss( inout HitInfo payload )
{
    payload.materialID = INVALID_ID;
}