package kernel

// #cgo CFLAGS: -DDEBUG

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/home/lizhen/CortexTheseus/infernet/build -lcvm_runtime -lcudart -lcuda
#cgo LDFLAGS: -lstdc++ 

#cgo CFLAGS: -I./include -I/usr/local/cuda/include/ -O2

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"fmt"
	"errors"
	"unsafe"
	"strings"
)

func LoadModel(modelCfg, modelBin string) (unsafe.Pointer, error) {
	net := C.CVMAPILoadModel(
		C.CString(modelCfg),
		C.CString(modelBin),
	)

	if net == nil {
		return nil, errors.New("Model load error")
	}
	return net, nil
}

func FreeModel(net unsafe.Pointer) {
	C.CVMAPIFreeModel(net)
}

func Predict(net unsafe.Pointer, imageData []byte) ([]byte, error) {
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}

	resLen := int(C.CVMAPIGetOutputLength(net))
	if resLen == 0 {
		return nil, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	flag := C.CVMAPIInfer(
		net,
		(*C.char)(unsafe.Pointer(&imageData[0])),
		(*C.char)(unsafe.Pointer(&res[0])))

	if flag != 0 {
		return nil, errors.New("Predict Error")
	}

	return res, nil
}

func InferCore(modelCfg, modelBin string, imageData []byte) ([]byte, error) {
	if (strings.Contains(strings.ToLower(modelCfg), "ca3d0286d5758697cdef653c1375960a868ac08a")) {
		modelCfg = "/tmp/ca3d_symbol";
		modelBin = "/tmp/ca3d_params";
	} else if (strings.Contains(strings.ToLower(modelCfg), "4d8bc8272b882f315c6a96449ad4568fac0e6038")) {
		return []byte{0}, nil;
	}
	fmt.Println(modelCfg, modelBin)
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
