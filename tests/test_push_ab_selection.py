"""A pilot may replace the parent only after success gains without old-skill loss."""
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_spring208_push_ab import meaningful, selection_score, restrict_to_wheel_head


def result():
    groups=('standing','accepted15','level','nominal_getup','difficult_getup')
    episodes=[dict(spec=dict(group=g,seed=i),success=i==0,warnings=0,seconds=10.)
        for g in groups for i in range(2)]
    return dict(episodes=episodes,summary={g:dict(success=1,count=2) for g in groups})


class SelectionTests(unittest.TestCase):
    def test_two_optimizers_cannot_change_frozen_leg_rows_or_features(self):
        actor=SimpleNamespace(standing_net=torch.nn.Linear(3,4),standing_head=torch.nn.Linear(4,6))
        policy=SimpleNamespace(mlp_extractor=SimpleNamespace(policy_net=actor))
        optimizers=[torch.optim.Adam(list(actor.standing_net.parameters())+list(actor.standing_head.parameters()),lr=.01)
                    for _ in range(2)]
        before=copy.deepcopy(actor.standing_head.state_dict())
        features=copy.deepcopy(actor.standing_net.state_dict())
        restrict_to_wheel_head(policy)
        for optimizer in optimizers*3:
            prediction=actor.standing_head(actor.standing_net(torch.ones(8,3)))
            optimizer.zero_grad(set_to_none=True)
            (prediction-10).square().mean().backward();optimizer.step()
        for key,value in actor.standing_net.state_dict().items():self.assertTrue(torch.equal(value,features[key]))
        for key,value in actor.standing_head.state_dict().items():
            self.assertTrue(torch.equal(value[:4],before[key][:4]))
            self.assertFalse(torch.equal(value[4:],before[key][4:]))

    def test_fixed_force_loss_rejects_random_band_gain(self):
        before=result();before['summary']['exact18']=dict(success=2,count=3)
        after=copy.deepcopy(before)
        after['summary']['level']['success']=2
        after['summary']['exact18']['success']=1
        self.assertFalse(meaningful(after,before))

    def test_fixed_force_success_gain_is_meaningful(self):
        before=result();before['summary']['exact18']=dict(success=1,count=3)
        after=copy.deepcopy(before);after['summary']['exact18']['success']=2
        self.assertTrue(meaningful(after,before))

    def test_fixed_force_success_precedes_random_band_score(self):
        before=result()
        before['summary']['exact18']=dict(success=2,count=3,recovered_push_count=2)
        before['summary']['level'].update(recovered_push_count=1)
        for row in before['episodes']:row['final_errors']=dict(position_m=0.,heading_deg=0.)
        after=copy.deepcopy(before)
        after['summary']['exact18']['success']=1
        after['summary']['level']['success']=2
        self.assertGreater(selection_score(before),selection_score(after))

    def test_survival_only_gain_is_not_accepted(self):
        before=result();after=copy.deepcopy(before)
        for e in after['episodes']:e['seconds']=20.
        self.assertFalse(meaningful(after,before))

    def test_new_success_with_retained_old_skills_is_eligible(self):
        before=result();after=copy.deepcopy(before)
        next(e for e in after['episodes'] if e['spec']==dict(group='level',seed=1))['success']=True
        after['summary']['level']['success']=2
        self.assertTrue(meaningful(after,before))

    def test_new_push_success_does_not_excuse_lost_standing(self):
        before=result();after=copy.deepcopy(before)
        next(e for e in after['episodes'] if e['spec']==dict(group='level',seed=1))['success']=True
        after['summary']['level']['success']=2
        next(e for e in after['episodes'] if e['spec']==dict(group='standing',seed=0))['success']=False
        after['summary']['standing']['success']=0
        self.assertFalse(meaningful(after,before))
