// Code generated by github.com/fjl/gencodec. DO NOT EDIT.

package torrentfs

import (
	"encoding/json"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
)

// MarshalJSON marshals as JSON.
func (r TxReceipt) MarshalJSON() ([]byte, error) {
	type Receipt struct {
		ContractAddr *common.Address `json:"ContractAddress"  gencodec:"required"`
		TxHash       *common.Hash    `json:"TransactionHash"  gencodec:"required"`
		GasUsed      hexutil.Uint64  `json:"gasUsed" gencodec:"required"`
		Status       hexutil.Uint64  `json:"status"`
	}
	var enc Receipt
	enc.ContractAddr = r.ContractAddr
	enc.TxHash = r.TxHash
	enc.GasUsed = hexutil.Uint64(r.GasUsed)
	enc.Status = hexutil.Uint64(r.Status)
	return json.Marshal(&enc)
}

// UnmarshalJSON unmarshals from JSON.
func (r *TxReceipt) UnmarshalJSON(input []byte) error {
	type Receipt struct {
		ContractAddr *common.Address `json:"ContractAddress"  gencodec:"required"`
		TxHash       *common.Hash    `json:"TransactionHash"  gencodec:"required"`
		GasUsed      hexutil.Uint64  `json:"gasUsed" gencodec:"required"`
		Status       hexutil.Uint64  `json:"status"`
	}
	var dec Receipt
	if err := json.Unmarshal(input, &dec); err != nil {
		return err
	}
	if dec.ContractAddr != nil {
		r.ContractAddr = dec.ContractAddr
	}
	if dec.TxHash != nil {
		r.TxHash = dec.TxHash
	}
	//if dec.GasUsed != nil {
	r.GasUsed = uint64(dec.GasUsed)
	//}
	//if dec.Status != nil {
	r.Status = uint64(dec.Status)
	//}
	return nil
}
