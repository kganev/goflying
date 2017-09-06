package magkal

import (
	"github.com/skelterjohn/go.matrix"
	"math"
	"fmt"
)

const (
	Big   = 1e4
	Small = 1e-4
)

// MagKalState contains the state variables for calibrating a magnetometer with a running Kalman filter.
// The matrix L is 3x4 and contains both the hard iron (offsets) and soft iron (squashing) corrections.
// N = L . M, where M is the raw magnetometer measurements (with a final 1 appended),
// L is the correction matrix, and N is the corrected output.
// N will be determined up to an overall rotation and rescaling, which must be determined by the consumer.
type MagKalState struct {
	L *matrix.DenseMatrix // Hard/Soft Correction Matrix, 12x1
	P *matrix.DenseMatrix // Uncertainty of L, 12x12
	Q *matrix.DenseMatrix // Process uncertainty, 12x12
	R *matrix.DenseMatrix // Measurement uncertainty, 1x1
	H *matrix.DenseMatrix // Measurement Jacobian, 1x12
	K *matrix.DenseMatrix // Kalman Gain, 12x1
	needsInitialization bool
	N float64 // Number of cycles processed
}

// Matrix returns the 3x4 transformation matrix corresponding to the current state.
func (s *MagKalState) Matrix() (ll *matrix.DenseMatrix) {
	return matrix.MakeDenseMatrix(s.L.Array(), 3, 4)
}

// MagKalMeasurement contains a measurement (or predicted measurement) of the external (fixed) magnetic field.
// It is a 1x1 matrix.
type MagKalMeasurement struct {
	M1, M2, M3 float64 // raw measured values
	T          float64 // time of measurement
}

// Norm returns the L2-norm of the measurement.
func (m *MagKalMeasurement) Norm() (n float64) {
	return math.Sqrt(m.M1*m.M1 + m.M2*m.M2 + m.M3*m.M3)
}

// Matrix returns the 4x1 matrix corresponding to the measurement.
func (m *MagKalMeasurement) Matrix() (mm *matrix.DenseMatrix) {
	return matrix.MakeDenseMatrix([]float64 {m.M1, m.M2, m.M3, 1}, 4, 1)
}

// NewMagKal returns a new Kalman Magnetometer calibrator.
func NewMagKal() (s *MagKalState) {
	s = new(MagKalState)

	s.L = matrix.Zeros(12, 1)
	s.P = matrix.Zeros(12, 12)

	s.Q = matrix.Diagonal([]float64 {
		Small, Small, Small, Big,
		Small, Small, Small, Big,
		Small, Small, Small, Big,
	})
	s.R = matrix.Diagonal([]float64 {1})

	s.H = matrix.Zeros(1, 12)
	s.K = matrix.Zeros(12, 1)

	s.needsInitialization = true
	return s
}

func (s *MagKalState) init(m *MagKalMeasurement) {
	mm := 1/m.Norm()
	s.L = matrix.MakeDenseMatrix([]float64 {
		mm, 0, 0, 0,
		0, mm, 0, 0,
		0, 0, mm, 0,
	},12, 1)

	mm = mm*mm
	s.P = matrix.Diagonal([]float64 {
		mm, mm*Small, mm*Small, mm*Big,
		mm*Small, mm, mm*Small, mm*Big,
		mm*Small, mm*Small, mm, mm*Big,
	})
	s.N++
}

// Compute runs just the update phase of the Kalman filter.
// Since the State doesn't change, the predict phase of the Kalman filter is trivial.
func (s *MagKalState) Compute(m *MagKalMeasurement) {
	if s.needsInitialization {
		s.init(m)
		return
	}

	// Predict phase of Kalman filter is trivial.
	mm := m.Matrix()
	nn := matrix.Product(s.Matrix(), mm)

	// Update the measurement noise:
	s.R.Set(0, 0,
		(s.N * s.R.Get(0, 0) + matrix.Product(nn.Transpose(), nn).Get(0, 0))/(s.N+1))

	// Update the state uncertainty estimate:
	s.P = matrix.Sum(s.P, s.Q)

	// Calculate the measurement Jacobian:
	for i:=0; i<3; i++ {
		for j:=0; j<4; j++ {
			s.H.Set(0, 4*i+j, 2 * mm.Get(0, j) * nn.Get(0, i))
		}
	}

	// Calculate the Kalman gain:
	s.K = matrix.Product(
		matrix.Product(s.P, s.H.Transpose(),
			matrix.Inverse(
				matrix.Sum(
					matrix.Product(s.H, s.P, s.H.Transpose()),
					s.R,
				),
			),
		),
	)

	// Correct the state estimate:
	s.L = matrix.Sum(
		s.L,
		matrix.Scaled(
			s.K,
			1 - matrix.Product(nn.Transpose(), nn).Get(0, 0),
		),
	)

	// Correct the state uncertainty estimate:
	cc := matrix.Sum(
		matrix.Eye(12),
		matrix.Scaled(matrix.Product(s.K, s.H), -1),
	)
	s.P = matrix.Sum(
		matrix.Product(cc, s.P, cc.Transpose()),
		matrix.Product(s.K, s.R, s.K.Transpose()),
	)

	s.N++
}

func (s *MagKalState) UpdateLogMap(m *MagKalMeasurement, logMap map[string]interface{}) {
	for i:=0; i<12; i++ {
		logMap[fmt.Sprintf("L%x", i+1)] = s.L.Get(i, 0) // Hard/Soft Correction Matrix, 12x1
		//logMap[fmt.Sprintf("H%d", i+1)] = s.H.Get(0, i) // Measurement Jacobian, 1x12
		//logMap[fmt.Sprintf("K%d", i+1)] = s.K.Get(i, 0) // Kalman Gain, 12x1
		//for j:=0; j<12; j++ {
		//	logMap[fmt.Sprintf("P%d%d", i+1, j+1)] = s.P.Get(i, j) // Uncertainty of L, 12x12
		//	logMap[fmt.Sprintf("Q%d%d", i+1, j+1)] = s.Q.Get(i, j) // Process uncertainty, 12x12
		//}
	}
	logMap["R11"] = s.R.Get(0, 0) // Measurement uncertainty, 1x1
	logMap["N"] = s.N
	nn := matrix.Product(s.Matrix(), m.Matrix())
	logMap["NN"] = matrix.Product(nn.Transpose(), nn).Get(0, 0)
}
