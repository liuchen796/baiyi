#include "balance_controller.h"
#include <math.h>

/* Sliding mode control parameters */
static float k_gain = 0.5f;      /* control gain */
static float c1 = 5.0f;          /* sliding surface angle gain */
static float c2 = 0.1f;          /* sliding surface rate gain */
static float target = 0.0f;      /* desired pitch angle */

void BalanceController_Init(float target_angle)
{
    target = target_angle;
    /* additional initialization can be added here */
}

static float satf(float s)
{
    const float sat_limit = 1.0f; /* simple saturation */
    if (s > sat_limit)
        return 1.0f;
    if (s < -sat_limit)
        return -1.0f;
    return s;
}

float BalanceController_Update(const IMU_Data_t *imu, float throttle_current)
{
    /* Compute sliding surface */
    float error = imu->angle - target;
    float s = c1 * error + c2 * imu->rate;

    /* Sliding mode control law */
    float control = -k_gain * satf(s);

    /* Combine with throttle command: controller output is the requested motor current */
    float output_current = throttle_current > control ? throttle_current : control;

    return output_current;
}
