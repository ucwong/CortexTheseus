package kernel

// #cgo CFLAGS: -DDEBUG

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -Wl,-rpath,./ -L./ -lcudart -lcvm_runtime
#cgo LDFLAGS: -lstdc++ 

#cgo CFLAGS: -I./include -I/usr/local/cuda/include/
#cgo CFLAGS: -DGPU -O3 -mavx2 -Wall

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

func LoadModel(modelCfg, modelBin string) (unsafe.Pointer, error) {
	net := C.cvm_load_model(
		C.CString(modelCfg),
		C.CString(modelBin),
	)

	if net == nil {
		return nil, errors.New("Model load error")
	}
	return net, nil
}

func FreeModel(net unsafe.Pointer) {
	C.cvm_free_model(net)
}

func Predict(net unsafe.Pointer, imageData []byte) ([]byte, error) {
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}

	resLen := int(C.cvm_get_output_length(net))
	if resLen == 0 {
		return nil, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	flag := C.cvm_predict(
		net,
		(*C.char)(unsafe.Pointer(&imageData[0])),
		(*C.char)(unsafe.Pointer(&res[0])))

	if flag != 0 {
		return nil, errors.New("Predict Error")
	}

	return res, nil
}

func InferCore(modelCfg, modelBin string, imageData []byte) ([]byte, error) {
	net, loadErr := LoadModel(modelCfg, modelBin)
	if loadErr != nil {
		return nil, errors.New("Model load error")
	}

	// Model load succeed
	defer FreeModel(net)

	return Predict(net, imageData)
	/*
		res, err := Predict(net, imageData)
		if err != nil {
			return 0, err
		}

		var (
			max    = int8(res[0])
			label  = uint64(0)
			resLen = len(res)
		)

		// If result length large than 1, find the index of max value;
		// Else the question is two-classify model, and value of result[0] is the prediction.
		if resLen > 1 {
			for idx := 1; idx < resLen; idx++ {
				if int8(res[idx]) > max {
					max = int8(res[idx])
					label = uint64(idx)
				}
			}
		} else {
			if max > 0 {
				label = 1
			} else {
				label = 0
			}
		}

		return label, nil */
}
