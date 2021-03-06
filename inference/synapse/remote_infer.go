package synapse

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/log"
	resty "gopkg.in/resty.v1"
)

func (s *Synapse) remoteGasByModelHash(modelInfoHash, uri string) (uint64, error) {
	inferWork := &inference.GasWork{
		Type:  inference.GAS_BY_H,
		Model: modelInfoHash,
	}

	requestBody, errMarshal := json.Marshal(inferWork)
	if errMarshal != nil {
		return 0, errMarshal
	}
	log.Debug("Remote Inference", "request", string(requestBody))

	retArray, err := s.sendRequest(string(requestBody), uri)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(retArray), nil
}

func (s *Synapse) remoteInferByInfoHash(modelInfoHash, inputInfoHash, uri string) ([]byte, error) {
	inferWork := &inference.IHWork{
		Type:  inference.INFER_BY_IH,
		Model: modelInfoHash,
		Input: inputInfoHash,
	}

	requestBody, err := json.Marshal(inferWork)
	if err != nil {
		return nil, err
	}
	log.Debug("Remote Inference", "request", string(requestBody))

	return s.sendRequest(string(requestBody), uri)
}

func (s *Synapse) remoteInferByInputContent(modelInfoHash, uri string, inputContent []byte) ([]byte, error) {
	inferWork := &inference.ICWork{
		Type:  inference.INFER_BY_IC,
		Model: modelInfoHash,
		Input: hexutil.Bytes(inputContent),
	}

	requestBody, err := json.Marshal(inferWork)
	if err != nil {
		return nil, err
	}
	log.Debug("Remote Inference", "request", string(requestBody))

	return s.sendRequest(string(requestBody), uri)
}

func (s *Synapse) sendRequest(requestBody, uri string) ([]byte, error) {
	cacheKey := RLPHashString(requestBody)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.([]byte))
		return v.([]byte), nil
	}

	resp, err := resty.R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(uri)
	if err != nil || resp.StatusCode() != 200 {
		return nil, errors.New(fmt.Sprintf("%s | %s | %s | %s | %v", "cvm.Infer: External Call Error: ", requestBody, resp, uri, err))
	}

	log.Debug("Remote Inference", "response", resp.String())

	var res inference.InferResult
	if jsErr := json.Unmarshal(resp.Body(), &res); jsErr != nil {
		return nil, errors.New(fmt.Sprintf("Remote Infer: resonse json parse error | %v ", jsErr))
	}

	if res.Info == inference.RES_OK {
		var data = []byte(res.Data)
		if !s.config.IsNotCache {
			s.simpleCache.Store(cacheKey, data)
		}
		return data, nil
	} else if res.Info == inference.RES_ERROR {
		return nil, errors.New(string(res.Data))
	}

	return nil, errors.New("Remote Infer: response json `info` parse error")
}
