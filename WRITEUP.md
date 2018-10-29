# Flying Car C++ Estimation Project Writeup

## 1. Sensor Noise

To complete this step we need to process log data and calculate std for GPS and accelerometer data.

```python
import numpy as np

gps_x = np.loadtxt('GPS.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]
acc_x = np.loadtxt('Accelerometer.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]

print(np.std(gps_x))
print(np.std(acc_x))
```

Resulting values are:

```python
MeasuredStdDev_GPSPosXY = 0.71
MeasuredStdDev_AccelXY = 0.49
```

![screen_1](images/scenario-1.png "Scenario 1")

## 2. Attitude Estimation

To complete this step we need to implement UpdateFromIMU() function. Integration scheme used here involves integrating gyro measurements in body frame and then converting them into global frame using quaternions.

Quaternions were chosen due to ease of implementation of IMU update procedure.

```cpp
void QuadEstimatorEKF::UpdateFromIMU(V3F accel, V3F gyro)
{
	bool smallAngle = false;
	float predictedPitch, predictedRoll;
	float gyroX, gyroY;

	Quaternion<float> qt, dq, pq;
	qt = qt.FromEuler123_RPY(rollEst, pitchEst, ekfState(6));
	dq = dq.IntegrateBodyRate(gyro, dtIMU);
	pq = dq * qt;
	
	predictedPitch = pq.Pitch(); // pitch
	predictedRoll = pq.Roll();   // roll
	ekfState(6) = pq.Yaw();     // yaw

	if (ekfState(6) > F_PI) ekfState(6) -= 2.f*F_PI;
	if (ekfState(6) < -F_PI) ekfState(6) += 2.f*F_PI;

	accelRoll = atan2f(accel.y, accel.z);
	accelPitch = atan2f(-accel.x, 9.81f);

	rollEst = attitudeTau / (attitudeTau + dtIMU) * (predictedRoll)+dtIMU / (attitudeTau + dtIMU) * accelRoll;
	pitchEst = attitudeTau / (attitudeTau + dtIMU) * (predictedPitch)+dtIMU / (attitudeTau + dtIMU) * accelPitch;

	lastGyro = gyro;
}
```

![screen_2](images/scenario-2.png "Scenario 2")

## 3. Prediction Step

To complete this step we need to implement GetRbgPrime() and PredictState() and Predict().

This scenario is about implementing prediction step of Kalman filter. GetRbgPrime calculates coodinate conversion matrix needed to properly adjust covariance matrix.

```cpp
VectorXf QuadEstimatorEKF::PredictState(VectorXf curState, float dt, V3F accel, V3F gyro)
{
	assert(curState.size() == QUAD_EKF_NUM_STATES);
	VectorXf predictedState = curState;

	Quaternion<float> attitude = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, curState(6));

	////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

	V3F accelGlobal = attitude.Rotate_BtoI(accel);
	accelGlobal[2] -= 9.81f;
	
	predictedState(0) += dt * ekfState(3);
	predictedState(1) += dt * ekfState(4);
	predictedState(2) += dt * ekfState(5);

	predictedState(3) += dt * accelGlobal[0];
	predictedState(4) += dt * accelGlobal[1];
	predictedState(5) += dt * accelGlobal[2];

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	return predictedState;
}

```

```cpp
void QuadEstimatorEKF::Predict(float dt, V3F accel, V3F gyro)
{
	VectorXf newState = PredictState(ekfState, dt, accel, gyro);

	MatrixXf RbgPrime = GetRbgPrime(rollEst, pitchEst, ekfState(6));

	MatrixXf gPrime(QUAD_EKF_NUM_STATES, QUAD_EKF_NUM_STATES);
	gPrime.setIdentity();

	gPrime(0, 3) = dt;
	gPrime(1, 4) = dt;
	gPrime(2, 5) = dt;
	
	MatrixXf _u(3, 1);
	_u(0, 0) = ekfState(0);
	_u(1, 0) = ekfState(1);
	_u(2, 0) = ekfState(2);

	MatrixXf _el = RbgPrime * _u;
	
	gPrime(3, 5) = _el(0, 0);
	gPrime(4, 5) = _el(1, 0);
	gPrime(5, 5) = _el(2, 0);

	ekfCov = gPrime * (ekfCov * gPrime.transpose()) + Q;

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	ekfState = newState;
}
```

```cpp
MatrixXf QuadEstimatorEKF::GetRbgPrime(float roll, float pitch, float yaw)
{
	MatrixXf RbgPrime(3, 3);
	RbgPrime.setZero();

	float sin_phi = sin(roll);
	float sin_theta = sin(pitch);
	float sin_psi = sin(yaw);

	float cos_phi = cos(roll);
	float cos_theta = cos(pitch);
	float cos_psi = cos(yaw);

	RbgPrime(0, 0) = -cos_theta * sin_psi;
	RbgPrime(0, 1) = -sin_phi * sin_theta * sin_psi - cos_phi * cos_psi;
	RbgPrime(0, 2) = -cos_phi * sin_theta * sin_psi + sin_phi * cos_psi;

	RbgPrime(1, 0) = cos_theta * sin_psi;
	RbgPrime(1, 1) = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi;
	RbgPrime(1, 2) = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi;

	return RbgPrime;
}
```

![screen_3](images/scenario-3.png "Scenario 3")

![screen_3_1](images/scenario-3-1.png "Scenario 3.1")

## 4. Magnetometer Update

To complete this step we need to implement UpdateFromMag() function end estimate magnetometer std dev.

```cpp
void QuadEstimatorEKF::UpdateFromMag(float magYaw)
{
	VectorXf z(1), zFromX(1);
	z(0) = magYaw;

	MatrixXf hPrime(1, QUAD_EKF_NUM_STATES);
	hPrime.setZero();

	////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

	hPrime(0, 6) = 1;
	zFromX(0) = ekfState(6);

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	Update(z, hPrime, R_Mag, zFromX);
}
```

![screen_4](images/scenario-4.png "Scenario 4")

## 5. Closed Loop + GPS Update

In order to complete this step we need to UpdateFromGPS() and tune GPS std dev in EKF estimator.

```cpp
void QuadEstimatorEKF::UpdateFromGPS(V3F pos, V3F vel)
{
	VectorXf z(6), zFromX(6);
	z(0) = pos.x;
	z(1) = pos.y;
	z(2) = pos.z;
	z(3) = vel.x;
	z(4) = vel.y;
	z(5) = vel.z;

	MatrixXf hPrime(6, QUAD_EKF_NUM_STATES);
	hPrime.setZero();

	for (int i = 0; i < 6; i++) {
		hPrime(i, i) = 1;
		zFromX(i) = ekfState(i);
	}

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	Update(z, hPrime, R_GPS, zFromX);
}
```

## 6. Adding Your Controller

In this step we need to add controller from previous project, revert to new estimator and realistic sensors and then tune parameters to successfully complete 11_GPSUpdate scenario.

Controller parameters adjusted for this scenario

```python
# Position control gains
kpPosXY = 12 # detuning to 12 didn't affect behaviour of controller in ideal conditions
kpPosZ = 35 # was detuned approx 30 percent
KiPosZ = 200 # I term of PID controller was increased 4x. Otherwise drone was getting off bneeded height and adjusted it slowly.

# Velocity control gains
kpVelXY = 10 # decreased 3x
kpVelZ = 40 # decreased 30%
```

![screen_5](images/scenario-4.png "Scenario 5")
