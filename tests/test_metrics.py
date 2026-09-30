import numpy as np
import pytest
from s1q.metrics import probabilities, metrics, compare, label_index, paired_accuracy_ci


def row(key,logits,label=0):
    return {"key":key,"record_id":key,"logits":logits,"label":label,"type":"choice","source":"fixture"}


def test_stable_softmax_and_brier_range():
    np.testing.assert_allclose(probabilities([1000,1000]),[.5,.5])
    m=metrics([row("a",[1000,-1000]),row("b",[-1000,1000])])
    assert m["accuracy"] == .5
    assert m["brier"] == 1
    assert m["coverage_at_risk"]["0.05"] == 0  # equal confidence tie cannot be cherry picked


def test_paired_keys_and_harmful_flip():
    ref=[row("a",[2,0]),row("b",[0,2])]
    quant=[row("b",[0,2]),row("a",[0,2])]
    result=compare(ref,quant)
    assert result["harmful_flip_rate"] == .5
    assert result["decision_flip_rate"] == .5
    assert paired_accuracy_ci(ref,quant,samples=100)["delta"] == -.5
    with pytest.raises(ValueError): compare(ref,quant[:1])


def test_label_contract():
    assert label_index({"type":"noul","label":True}) == 1
    assert label_index({"type":"choice","criteria":{"b":"B","a":"A"},"label":"a"}) == 1
    with pytest.raises(ValueError): probabilities([float("nan"),0])
