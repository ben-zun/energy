from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition,Element

from analysis import ROOT,EXP,OUT,digest


def fractions(formulas):
    return np.array([[Composition(str(formula)).get_atomic_fraction(Element.from_Z(number)) for number in range(1,119)] for formula in formulas])


def main():
    archive_path=ROOT/'runs/rclfa_v3/eqv3_matched_benchmark_20260725/calibration/calibrated_predictions.csv'
    archive=pd.read_csv(archive_path,keep_default_na=False)
    coefficients=np.load(EXP/'eqv3_fitted_offset_coefficients.npz')
    weight=coefficients['w']; bias=float(coefficients['b'])
    features=fractions(archive.formula)
    reconstructed=features@weight+bias
    original=pd.read_csv(EXP/'eqv3_fitted_reference.csv',keep_default_na=False).set_index('id').loc[archive.id,'pred_element_offset_fitted'].to_numpy()
    np.testing.assert_allclose(reconstructed,original,atol=1e-10,rtol=0)
    training=archive.split.eq('train').to_numpy()
    archived_fit_residual=(archive.y_true-archive.model_energy_per_atom).to_numpy()-reconstructed
    centered_features=features[training]-features[training].mean(axis=0)
    gradient=centered_features.T@archived_fit_residual[training]-1e-6*weight
    archived_fit_stationary=float(np.max(np.abs(gradient)))<1e-5 and abs(archived_fit_residual[training].mean())<1e-8
    if not archived_fit_stationary:
        raise AssertionError('Stored coefficients not reproduced by archived train-only ridge normal equations')
    live_train=pd.read_csv(OUT/'support/initialization_rows_train.csv')
    live_train=live_train[live_train['mode'].eq('eval')]
    train_features=fractions(archive.set_index('id').loc[live_train.id].formula)
    live_residual=live_train.truth_ev_atom.to_numpy()-live_train.native_direct_ev_atom.to_numpy()-(train_features@weight+bias)
    live_gradient=(train_features-train_features.mean(axis=0)).T@live_residual-1e-6*weight
    current_init_fit_valid=abs(live_residual.mean())<1e-6 and np.max(np.abs(live_gradient))<1e-3
    rows=[]; diagnostic={}
    for split in ('train','val'):
        frame=pd.read_csv(OUT/'support'/f'initialization_rows_{split}.csv')
        state=json.loads((OUT/'support'/f'initialization_state_{split}.json').read_text())
        beta=state['beta']; normalization=state['normalization']
        info=archive.set_index('id').loc[frame.id]
        atoms=frame.natoms.to_numpy()
        xvalues=fractions(info.formula)
        element_refs=np.array(normalization['element_references'])[1:119]
        physical=frame.native_direct_ev_atom.to_numpy()*normalization['energy_rmsd']+normalization['energy_mean']/atoms+xvalues@element_refs
        gap=np.abs(frame.native_direct_ev_atom-frame.archive_energy_ev_atom)
        reconstructed_gap=np.abs(physical-frame.archive_energy_ev_atom.to_numpy())
        groups=['|'.join(map(str,row)) for row in np.rint(xvalues*atoms[:,None]).astype(int)]
        spread=pd.DataFrame({'group':groups,'offset':frame.offset_ev_atom}).groupby('group').offset.agg(lambda values:values.max()-values.min()).max()
        identity=float(np.max(np.abs(frame.native_clean_ev_atom-frame.native_direct_ev_atom-frame.offset_ev_atom)))
        diagnostic[split]={'native_vs_archive_max_ev_atom':float(gap.max()),
            'native_vs_archive_mae_ev_atom':float(gap.mean()),
            'denormalized_native_vs_archive_mae_ev_atom':float(reconstructed_gap.mean()),
            'denormalized_native_vs_archive_max_ev_atom':float(reconstructed_gap.max()),
            'offset_within_group_max_ev_atom':float(spread),'clean_minus_direct_identity_max_ev_atom':identity,
            'archive_filled_prediction_unique_ids':int(info[info.primary_missing.astype(float)>0].index.nunique()),
            'denormalized_native_vs_archive_median_ev_atom':float(np.median(reconstructed_gap))}
        for mode,block in frame.groupby('mode'):
            for route in ('direct','clean'):
                error=block[f'native_{route}_ev_atom']-block.truth_ev_atom
                residual=block[f'{route}_loss_residual']
                proxy=block.archive_energy_ev_atom+(block.offset_ev_atom if route=='clean' else 0)
                manual=np.where(abs(residual)<beta,.5*residual**2/beta,abs(residual)-.5*beta)
                np.testing.assert_allclose(manual,block[f'{route}_loss'],rtol=1e-5,atol=2e-5)
                rows.append({'route':route,'split':split,'mode':mode,'seed':20260528,'n':len(block),
                    'native_mae_mev_atom':1000*float(abs(error).mean()),'native_rmse_mev_atom':1000*float(np.sqrt(np.mean(error**2))),
                    'actual_smooth_l1':float(block[f'{route}_loss'].mean()),'residual_mean_ev_atom':float(error.mean()),
                    'loss_residual_mean_scaled_total':float(residual.mean()),'saturated_fraction':float((abs(residual)>=beta).mean()),
                    'huber_beta_scaled_total':beta,'scale_divisor':state['route_states'][0]['scale'],
                    'weights_equal_between_routes':state['route_states'][0]['weight_hash']==state['route_states'][1]['weight_hash'],
                    'output_heads_equal':state['route_states'][0]['head_hash']==state['route_states'][1]['head_hash'],
                    'lora_B_zero':all(item['lora_B_abs_max']==0 for item in state['route_states']),
                    'no_parameter_updates':state['no_parameter_updates'],
                    'reference_train_only_archive_fit_verified':archived_fit_stationary,
                    'reference_fitted_to_native_initialization':bool(current_init_fit_valid),
                    'archive_filled_predictions_in_split':diagnostic[split]['archive_filled_prediction_unique_ids'],
                    'archive_proxy_mae_mev_atom':1000*float(abs(proxy-block.truth_ev_atom).mean()),
                    'offset_group_spread_ev_atom':float(spread),'identity_max_ev_atom':identity,
                    'initial_checkpoint':state['config']['checkpoint'],
                    'reference_csv':str(EXP/'eqv3_fitted_reference.csv'),'reference_sha256':digest(EXP/'eqv3_fitted_reference.csv')})
    pd.DataFrame(rows).to_csv(OUT/'initialization_audit.csv',index=False)
    result={'archived_train_fit_normal_equation_max':float(np.max(np.abs(gradient))),
        'archive_ridge_intercept_residual':float(archived_fit_residual[training].mean()),
        'live_train_intercept_residual_ev_atom':float(live_residual.mean()),
        'live_train_normal_equation_max':float(np.max(np.abs(live_gradient))),
        'current_init_fit_valid':bool(current_init_fit_valid),
        'reference_was_not_refit_or_overwritten':True,'mappings':diagnostic,
        'normalization_explanation':'archive e0 was denormalized with energy rmsd and atomic element references; native trainer uses un-denormalized network energy directly',
        'validity_gate':'failed: clean offset was not fitted against actual native initialization output',
        'new_training_decision':'do not launch composition-offset training until initialization/reference contract is repaired and audited separately'}
    (OUT/'support/initialization_contract.json').write_text(json.dumps(result,indent=2))
    print(pd.DataFrame(rows)[['route','split','mode','native_mae_mev_atom','actual_smooth_l1','saturated_fraction']].to_string(index=False))


if __name__=='__main__':
    main()
