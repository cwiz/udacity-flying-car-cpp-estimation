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

## 2. Attitude Estimation

To complete this step we need to implement UpdateFromIMU() function. Integration scheme used here involves integrating gyro measurements in body frame and then converting them into global frame using quaternions.

```cpp
void QuadEstimatorEKF::UpdateFromIMU(V3F accel, V3F gyro)
{
	bool smallAngle = false;
	float predictedPitch, predictedRoll;
	float gyroX, gyroY;

	// This scenario is only viable for small angles where we can safely neglect coordinate system conversions
	if (smallAngle) {
		gyroX = gyro.x;
		gyroY = gyro.y;	

		predictedPitch = pitchEst + dtIMU * gyroY;  // pitch
		predictedRoll = rollEst + dtIMU * gyroX;    // roll
		ekfState(6) = ekfState(6) + dtIMU * gyro.z;	// yaw
	} else {
		Quaternion<float> qt, dq, pq;
		qt = qt.FromEuler123_RPY(rollEst, pitchEst, ekfState(6));
		dq = dq.IntegrateBodyRate(gyro, dtIMU);
		pq = dq * qt;
		
		predictedPitch = pq.Pitch(); // pitch
		predictedRoll = pq.Roll();   // roll
		ekfState(6) = pq.Yaw();     // yaw
	}
	
	// normalize yaw to -pi .. pi
	if (ekfState(6) > F_PI) ekfState(6) -= 2.f*F_PI;
	if (ekfState(6) < -F_PI) ekfState(6) += 2.f*F_PI;

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	// CALCULATE UPDATE
	accelRoll = atan2f(accel.y, accel.z);
	accelPitch = atan2f(-accel.x, 9.81f);

	// FUSE INTEGRATION AND UPDATE
	rollEst = attitudeTau / (attitudeTau + dtIMU) * (predictedRoll)+dtIMU / (attitudeTau + dtIMU) * accelRoll;
	pitchEst = attitudeTau / (attitudeTau + dtIMU) * (predictedPitch)+dtIMU / (attitudeTau + dtIMU) * accelPitch;

	lastGyro = gyro;
}
```

## 3. Prediction Step

To complete this step we need to implement GetRbgPrime() and Predict().

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

	// Rbg Prime
	RbgPrime(0, 0) = -cos_theta * sin_psi;
	RbgPrime(0, 1) = -sin_phi * sin_theta * sin_psi - cos_phi * cos_psi;
	RbgPrime(0, 2) = -cos_phi * sin_theta * sin_psi + sin_phi * cos_psi;

	RbgPrime(1, 0) = cos_theta * sin_psi;
	RbgPrime(1, 1) = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi;
	RbgPrime(1, 2) = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi;


	/////////////////////////////// END STUDENT CODE ////////////////////////////

	return RbgPrime;
}
```

```cpp
void QuadEstimatorEKF::Predict(float dt, V3F accel, V3F gyro)
{
	// predict the state forward
	VectorXf newState = PredictState(ekfState, dt, accel, gyro);

	// we'll want the partial derivative of the Rbg matrix
	MatrixXf RbgPrime = GetRbgPrime(rollEst, pitchEst, ekfState(6));

	// we've created an empty Jacobian for you, currently simply set to identity
	MatrixXf gPrime(QUAD_EKF_NUM_STATES, QUAD_EKF_NUM_STATES);
	gPrime.setIdentity();

	////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

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

	// GPS UPDATE
	// Hints: 
	//  - The GPS measurement covariance is available in member variable R_GPS
	//  - this is a very simple update
	////////////////////////////// BEGIN STUDENT CODE ///////////////////////////

	hPrime(0, 0) = 1.f;
	hPrime(1, 1) = 1.f;
	hPrime(2, 2) = 1.f;
	hPrime(3, 3) = 1.f;
	hPrime(4, 4) = 1.f;
	hPrime(5, 5) = 1.f;

	zFromX(0) = ekfState(0);
	zFromX(1) = ekfState(1);
	zFromX(2) = ekfState(2);
	zFromX(3) = ekfState(3);
	zFromX(4) = ekfState(4);
	zFromX(5) = ekfState(5);

	/////////////////////////////// END STUDENT CODE ////////////////////////////

	Update(z, hPrime, R_GPS, zFromX);
}
```

## 6. Adding Your Controller

In this step we need to add controller from previous project and then tune parameters to successfully complete 11_GPSUpdate scenario.