#include "balance_controller.h"
#include "stm32f4xx_hal.h"

/* Placeholder IMU read function */
extern void IMU_Read(IMU_Data_t *data);
/* Placeholder motor driver */
extern void Motor_SetCurrent(float current);
/* Placeholder throttle reading */
extern float GetThrottleCurrent(void);

int main(void)
{
    HAL_Init();
    /* System clock and peripheral initialization should be done here */

    BalanceController_Init(20.0f); /* target pitch angle is 20 degrees */

    while (1)
    {
        IMU_Data_t imu;
        IMU_Read(&imu);

        float throttle = GetThrottleCurrent();
        float current = BalanceController_Update(&imu, throttle);
        Motor_SetCurrent(current);
    }
}
