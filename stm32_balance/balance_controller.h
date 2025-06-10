#ifndef BALANCE_CONTROLLER_H
#define BALANCE_CONTROLLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float angle;      /* Current pitch angle in degrees */
    float rate;       /* Current pitch rate (d/dt) in deg/s */
} IMU_Data_t;

void BalanceController_Init(float target_angle);
float BalanceController_Update(const IMU_Data_t *imu, float throttle_current);

#ifdef __cplusplus
}
#endif

#endif /* BALANCE_CONTROLLER_H */
