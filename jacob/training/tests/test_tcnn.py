"""Geometry, causality, data isolation and recovery tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader

from baseline_pipeline import geodesic_degrees
from smoke_tcnn import make_fixture
from tcnn.data import check_corpus, WindowDataset
from tcnn.model import CausalBlock, TinyTCNN, conjugate, qmul, rotation_loss, rotvec_to_quaternion
import train_tcnn

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def test_composition_matches_scipy_noncommuting(self):
        a = Rotation.from_euler("x", 40, degrees=True)
        b = Rotation.from_euler("z", -25, degrees=True)
        qa = torch.tensor(a.as_quat()[[3,0,1,2]])
        qb = torch.tensor(b.as_quat()[[3,0,1,2]])
        np.testing.assert_allclose(qmul(qa, qb), (a*b).as_quat()[[3,0,1,2]], atol=1e-12)
        np.testing.assert_allclose(qmul(conjugate(qa),qa), [1,0,0,0], atol=1e-12)

    def test_rotation_vectors_and_zero_gradient(self):
        v = torch.tensor([[0.,0.,0.], [.2,-.1,.5]], dtype=torch.float64, requires_grad=True)
        q = rotvec_to_quaternion(v)
        expected = Rotation.from_rotvec(v.detach()).as_quat()[:, [3,0,1,2]]
        np.testing.assert_allclose(q.detach(), expected, atol=1e-12)
        q[:,1:].sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        torch.testing.assert_close(v.grad[0], torch.full((3,), .5, dtype=torch.float64))

    def test_loss_sign_invariance(self):
        a = rotvec_to_quaternion(torch.randn(3,20,3))
        b = rotvec_to_quaternion(torch.randn(3,20,3))
        torch.testing.assert_close(rotation_loss(a,b), rotation_loss(-a,b))

    def test_model_contract_and_hold_initialization(self):
        model = TinyTCNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 26972)
        q = rotvec_to_quaternion(torch.randn(2,400,3)*.1)
        pred = model(q)
        self.assertEqual(tuple(pred.shape), (2,20,4))
        np.testing.assert_allclose(geodesic_degrees(pred.detach(), q[:,-1:, :].expand(-1,20,-1)), 0, atol=1e-4)
        torch.testing.assert_close(pred.norm(dim=-1), torch.ones(2,20))

    def test_world_rotation_equivariance_and_signs(self):
        model = TinyTCNN(dropout=0).eval()
        torch.nn.init.normal_(model.head.weight, std=.01)
        q = rotvec_to_quaternion(torch.randn(2,400,3)*.1)
        world = rotvec_to_quaternion(torch.tensor([.6,-.7,.3]))
        expected = qmul(world, model(q))
        actual = model(qmul(world,q))
        np.testing.assert_allclose(geodesic_degrees(expected.detach(),actual.detach()),0,atol=1e-4)
        q[:,::3] *= -1
        np.testing.assert_allclose(geodesic_degrees(qmul(world,model(q)).detach(),actual.detach()),0,atol=1e-4)

    def test_temporal_block_never_reads_future_features(self):
        block = CausalBlock(4, 8, 0).eval()
        x = torch.randn(2,4,80)
        changed = x.clone()
        changed[:,:,40:] = torch.randn(2,4,40)*100
        torch.testing.assert_close(block(x)[:,:,:40], block(changed)[:,:,:40], rtol=0, atol=0)

    def test_receptive_field_reaches_first_input(self):
        model = TinyTCNN(dropout=0).eval()
        for layer in (model.projection, *(b.conv for b in model.blocks)):
            torch.nn.init.constant_(layer.weight, .01)
            torch.nn.init.constant_(layer.bias, .1)
        x = torch.ones(1,4,400,requires_grad=True)
        model.blocks(torch.relu(model.projection(x)))[:,:,-1].sum().backward()
        self.assertGreater(float(x.grad[:,:,0].abs().sum()), 0)

    def test_can_overfit_32_motion_windows(self):
        torch.manual_seed(9)
        model = TinyTCNN(dropout=0)
        rates = torch.linspace(.4,1.0,32)[:,None]
        angles = rates * torch.arange(-399,21)[None,:]/100
        rv = torch.zeros(32,420,3)
        rv[:,:,2] = angles
        q = rotvec_to_quaternion(rv)
        x,y = q[:,:400],q[:,400:]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        initial = float(rotation_loss(model(x),y).detach())
        for _ in range(100):
            optimizer.zero_grad()
            loss = rotation_loss(model(x),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
        final = float(rotation_loss(model(x),y).detach())
        self.assertLess(final, initial*.2, (initial,final))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data,self.split = make_fixture(self.directory / "fixture")

    def tearDown(self):
        self.temp.cleanup()

    def audit(self):
        return check_corpus(self.data,self.split,minimum_users=3,minimum_recordings=3,synthetic=True)

    def test_real_training_rejects_synthetic_and_small_corpus(self):
        with self.assertRaisesRegex(ValueError,"Synthetic"):
            check_corpus(self.data,self.split)
        with self.assertRaisesRegex(ValueError,"below training minimum"):
            check_corpus(self.data,self.split,synthetic=True)

    def test_loader_targets_are_future_and_training_users_only(self):
        audit = self.audit()
        rows = [r for r in audit["segments"] if r["split"] == "train"]
        dataset = WindowDataset(self.data, rows, 20, 42)
        try:
            self.assertEqual(dataset.users,["synthetic_train"])
            with h5py.File(self.data) as h5:
                full = h5[rows[0]["path"]]["quaternion_wxyz"][:]
            x,y = dataset[0]
            candidates = np.flatnonzero(np.all(np.isclose(full,x[0].numpy(),atol=1e-7),axis=1))
            self.assertEqual(len(candidates),1)
            start = candidates[0]
            np.testing.assert_array_equal(x,full[start:start+400])
            np.testing.assert_array_equal(y,full[start+400:start+420])
            torch.testing.assert_close(x,dataset[0][0])
        finally:
            dataset.close()

    def test_windows_worker_spawn_matches_single_process(self):
        rows = [r for r in self.audit()["segments"] if r["split"] == "train"]
        dataset = WindowDataset(self.data,rows,8,42)
        expected = torch.stack([dataset[i][0] for i in range(8)])
        dataset.close()
        loader = DataLoader(dataset,batch_size=4,num_workers=2,multiprocessing_context="spawn")
        actual = torch.cat([x for x,y in loader])
        torch.testing.assert_close(actual,expected)
        dataset.close()

    def test_manifest_split_must_match_the_file_groups(self):
        # The HDF5's /<split> groups are authoritative; a manifest that disagrees is refused.
        manifest = json.loads(self.split.read_text())
        manifest["splits"]["train"], manifest["splits"]["validation"] = manifest["splits"]["validation"], manifest["splits"]["train"]
        self.split.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,"differ from the HDF5"):
            self.audit()

    def test_data_checksum_change_detected(self):
        with h5py.File(self.data,"r+") as h5:
            h5.attrs["changed_after_freezing"] = True
        with self.assertRaisesRegex(ValueError,"checksum"):
            self.audit()

    def test_sampling_balances_users_with_unequal_recording_lengths(self):
        path = self.directory / "unequal.h5"
        rows = []
        with h5py.File(path,"w") as h5:
            for name,n,identity in [("short",500,[1,0,0,0]),("long",20000,[0,1,0,0])]:
                key = f"{name}/recording/segment_000"
                h5.create_dataset(key+"/quaternion_wxyz",data=np.tile(np.float32(identity),(n,1)))
                rows.append({"user":name,"key":key,"path":key,"samples":n})
        dataset = WindowDataset(path,rows,1000,42)
        try:
            short = sum(float(dataset[i][0][0,0]) == 1 for i in range(1000))
            self.assertGreater(short,400)
            self.assertLess(short,600)
        finally:
            dataset.close()

    def test_resume_matches_uninterrupted_training(self):
        config = json.loads((ROOT / "configs/tcnn_3080.json").read_text())
        config.update(batch_size=4, steps_per_epoch=2, epochs=2,workers=0,validation_stride=20)
        full = self.directory / "full"
        resumed = self.directory / "resumed"
        train_tcnn.train(self.data,self.split,full,config,42,"cpu",synthetic=True)
        original = train_tcnn.atomic_checkpoint
        def stop_after_saved_epoch(path,state):
            original(path,state)
            if Path(path).name == "last.pt" and state["epoch"] == 1:
                raise InterruptedError("simulated interruption")
        with patch.object(train_tcnn,"atomic_checkpoint",side_effect=stop_after_saved_epoch):
            with self.assertRaises(InterruptedError):
                train_tcnn.train(self.data,self.split,resumed,config,42,"cpu",synthetic=True)
        train_tcnn.train(self.data,self.split,resumed,config,42,"cpu",resume=True,synthetic=True)
        a = torch.load(full / "last.pt",weights_only=True)
        b = torch.load(resumed / "last.pt",weights_only=True)
        for name in a["model_state"]:
            torch.testing.assert_close(a["model_state"][name],b["model_state"][name],rtol=0,atol=0)
        self.assertEqual(a["best_validation_deg"],b["best_validation_deg"])
        changed = dict(config,epochs=3)
        with self.assertRaisesRegex(ValueError,"identical configuration"):
            train_tcnn.train(self.data,self.split,resumed,changed,42,"cpu",resume=True,synthetic=True)


if __name__ == "__main__":
    unittest.main()
