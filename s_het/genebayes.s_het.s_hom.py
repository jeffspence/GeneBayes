import pickle
import numpy as np
import pandas as pd
import argparse

import xgboost as xgb
from ngboost import NGBRegressor
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

import torch
import torch.distributions as dist


def integrate(these_params, these_likelihoods, logit=True):
    result = 0

    grid = torch.linspace(-1., 1., N_INTEGRATION_PTS).expand(1, -1)
    means_s_het, means_s_hom = Prior.get_mean(these_params)
    sds_s_het, sds_s_hom = Prior.get_sd(these_params)
    grid_s_het = (
        means_s_het.expand(-1, 1) + grid * 8 * sds_s_het.expand(-1, 1)
    )
    grid_s_hom = (
        means_s_hom.expand(-1, 1) + grid * 8 * sds_s_hom.expand(-1, 1)
    )

    tile_grid_s_het = grid_s_het.repeat_interleave(len(grid_s_hom))
    tile_grid_s_hom = grid_s_hom.repeat(len(grid_s_het))

    eval_at_pts, prior_p = log_prob(Prior.distribution,
                                    these_params,
                                    these_likelihoods,
                                    tile_grid_s_het,
                                    tile_grid_s_hom,
                                    logit=logit)

    eval_at_pts = eval_at_pts.reshape(-1, len(grid_s_het), len(grid_s_hom))

    m = torch.max(
        torch.max(eval_at_pts, dim=1, keepdim=True)[0],
        dim=2,
        keepdim=True
    )[0]
    pdf = eval_at_pts - m

    result += torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf), grid_s_hom, dim=2
        ),
        grid_s_het,
        dim=1
    )

    m = torch.max(
        torch.max(prior_p, dim=1, keepdim=True)[0],
        dim=2,
        keepdim=True
    )[0]

    check = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(prior_p - m), grid_s_hom, dim=2
        ),
        grid_s_het,
        dim=1
    )

    if np.any(np.abs(check.detach().cpu().numpy() - 1.) > 1e-5):
        bad_idx = np.argmax(np.abs(check.detach().cpu().numpy()-1.))
        print('Discrepancy of ',
              np.max(np.abs(check.detach().cpu().numpy() - 1.)),
              'found while integrating')
        print('params were', [p[bad_idx] for p in these_params])
        print(torch.exp(prior_p[bad_idx, 0]), torch.exp(prior_p[bad_idx, -1]))
        print('mean was', means_s_het[bad_idx], means_s_hom[bad_idx])
        print('sd was', sds_s_het[bad_idx], sds_s_hom[bad_idx])
        print('grid was', grid[bad_idx])

    return result


class PriorScore(LogScore):
    hessian = None

    def score(self, y, compute_grad=False):
        """
        log score, -log P(y), for prior distribution P
        """
        params = torch.tensor(self.params_transf, device=DEVICE)
        params = params.unsqueeze(dim=-1)
        params.requires_grad = True

        score = 0
        if compute_grad:
            grad = np.zeros((self.n_params, y.shape[0]))

            # Just for debugging
            # grad_param_plus = np.zeros(
            #     (self.n_params, self.n_params, y.shape[0])
            # )

            hessian = np.zeros(
                (y.shape[0], self.n_params, self.n_params)
            )
        y = torch.tensor(y, device=DEVICE)

        for idx in torch.split(
            torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
        ):
            idx_numpy = idx.cpu().numpy()
            these_params = torch.tensor(
                self.params_transf[:, idx_numpy], device=DEVICE
            ).unsqueeze(dim=-1)

            if compute_grad:
                these_params.requires_grad = True

            result = integrate(these_params, y[idx])
            result = -torch.sum(torch.log(result)+torch.max(y[idx], dim=1)[0])

            if compute_grad:
                this_grad = torch.autograd.grad(
                    result,
                    these_params,
                    retain_graph=True,
                    create_graph=True
                )[0][:, :, 0]
                grad[:, idx_numpy] = [
                    tg.detach().squeeze().cpu().numpy() for tg in this_grad
                ]

            score += result.item()

            # This for loop is just for debugging and computes stuff needed to
            # numerically compute the hessian
            '''
            for perturb_idx in range(self.n_params):
                perturb_params = torch.tensor(
                    self.params_transf[:, idx_numpy], device=DEVICE
                ).unsqueeze(dim=-1)
                perturb_params[perturb_idx] += 1e-6
                perturb_params = perturb_params.detach().clone()
                perturb_params.requires_grad = True
                result = integrate(perturb_params, y[idx])
                result = -torch.sum(torch.log(result))
                result.backward()
                grad_param_plus[perturb_idx, :, idx_numpy] = (
                    perturb_params.grad.squeeze().T.cpu().numpy()
                )
            '''
            if compute_grad:
                for p_idx in range(self.n_params):
                    torch.sum(this_grad[p_idx, :]).backward(retain_graph=True)
                    this_hess_row = (
                        these_params.grad.squeeze().T.detach().cpu().numpy()
                    )
                    hessian[idx_numpy, p_idx, :] = this_hess_row
                    these_params.grad.zero_()

        # This is just for debugging, check that hessian is close to numerical
        # hessian
        '''
        numerical_hessian = np.zeros_like(hessian)
        for p_plus in range(self.n_params):
            for p in range(self.n_params):
                numerical_hessian[:, p_plus, p] = (
                    (grad_param_plus[p_plus, p, :] - grad[p, :])
                    / 1e-6
                )

        print('First gradient is')
        print(grad[:, 0])
        print('First hessian is')
        print(hessian[0])
        print('Numerical hessian is')
        print(numerical_hessian[0])

        abs_diff = np.abs(hessian - numerical_hessian)
        error = abs_diff.sum(axis=(1, 2))
        print('Worst hessians:', np.argmax(error))
        print(hessian[np.argmax(error)])
        print(numerical_hessian[np.argmax(error)])
        avg_hess = 0.5 * (np.abs(hessian) + np.abs(numerical_hessian))
        print('worst absolute difference is', np.max(abs_diff))
        print('worst relative difference is',
              np.max(abs_diff / (avg_hess + 1e-5)))
        '''
        # Ensure symmetry
        if compute_grad:
            if not np.allclose(hessian, np.transpose(hessian, [0, 2, 1])):
                print(
                    'Encountered asymmetric hessian with maximal deviance of',
                    np.max(np.abs(hessian - np.transpose(hessian, [0, 2, 1])))
                )

            hessian = 0.5 * (hessian + np.transpose(hessian, [0, 2, 1]))

            for p1, p1_pos in enumerate(Prior.positive):
                for p2, p2_pos in enumerate(Prior.positive):
                    if p1 == p2 and p1_pos:
                        hessian[:, p1, p2] = (
                            hessian[:, p1, p2] * self.params_transf[p1, :]**2
                            + grad[p1, :] * self.params_transf[p1, :]
                        )
                        continue
                    if p1_pos:
                        hessian[:, p1, p2] *= self.params_transf[p1, :]
                    if p2_pos:
                        hessian[:, p1, p2] *= self.params_transf[p2, :]

            assert not np.any(np.isnan(hessian))
            assert np.allclose(hessian, np.transpose(hessian, [0, 2, 1]))
            self.hessian = hessian

            untransf_grad = np.copy(grad)
            untransf_grad[Prior.positive] *= (
                self.params_transf[Prior.positive, :]
            )
            self.gradient = np.copy(untransf_grad.T)

        return score

    def d_score(self, data):
        """
        derivative of the score
        """
        for i in range(Prior.n_params):
            p0 = np.min(self.params_transf[i])
            p1 = np.quantile(self.params_transf[i], 0.01)
            p99 = np.quantile(self.params_transf[i], 0.99)
            p100 = np.max(self.params_transf[i])
            print(
                f"param #{i} - min: {p0}, 1st percentile: {p1}, "
                f"99th percentile: {p99}, max: {p100}"
            )

        self.score(data, compute_grad=True)

        return np.copy(self.gradient)

    def metric(self):
        return self.regularized_hessian()

    def regularized_hessian(self):
        print('Mean, percentiles of grad norms',
              np.mean(np.sqrt((self.gradient**2).sum(axis=1))),
              np.percentile(np.sqrt((self.gradient**2).sum(axis=1)),
                            [0, 25, 50, 75, 100]))

        regularized_hessian = np.copy(self.hessian)
        evals, evecs = np.linalg.eigh(self.hessian)
        print('Mean, percentiles of eigenvalues', np.mean(np.abs(evals)),
              np.percentile(np.abs(evals), [0, 25, 50, 75, 100]))
        clipped_evals = np.clip(
            np.abs(evals),
            HESS_REG_1*np.mean(np.abs(evals)),
            # this_reg_val,
            float('inf')
        )
        print('After clipping, Mean, percentiles of eigenvalues',
              np.mean(np.abs(clipped_evals)),
              np.percentile(np.abs(clipped_evals), [0, 25, 50, 75, 100]))

        regularized_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            clipped_evals,
            evecs
        ) + HESS_REG_2*np.mean(np.abs(evals))*np.eye(self.n_params)[None, :, :]

        # The below is just for debugging and checking stuff out.
        '''
        diff = np.abs(evals.min(axis=1)) * (evals.min(axis=1) < 0)
        scale = np.abs(evals).max(axis=1)
        reg = diff + 1e-3*scale + 0.1
        reg = reg[:, None, None]
        print(self.hessian[0])
        print(self.gradient[0], 'Raw gradient')
        this_approx = np.linalg.solve(self.hessian, self.gradient)
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'raw_hessian')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        this_approx = np.linalg.solve(
            self.hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'Pure regularization')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        clipped_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            np.clip(np.abs(evals), 0., float('inf')),
            evecs
        )
        this_approx = np.linalg.solve(
            clipped_hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'Clip eigenvals + reg')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        clipped_hessian = np.einsum(
            'nij,nj,nkj->nik',
            evecs,
            np.clip(np.abs(evals), 0.1, float('inf')),
            evecs
        )
        this_approx = np.linalg.solve(
            clipped_hessian + reg * np.eye(self.n_params)[None, :, :],
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'reg in eigenspace')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        this_approx = np.linalg.solve(
            regularized_hessian,
            self.gradient
        )
        this_approx_norms = np.sqrt((this_approx**2).sum(axis=1))
        print(this_approx[0], 'actual regularization')
        print('Mean, percentiles of grad norms',
              np.mean(this_approx_norms),
              np.percentile(this_approx_norms, [0, 25, 50, 75, 100]))

        '''

        return regularized_hessian


def log_prob(
    prior_dist, these_params, these_likelihoods, s_het, s_hom, logit=True
):
    prior_p = prior_dist(these_params, logit).log_prob(torch.tensor([s_het,
                                                                     s_hom]))

    if logit:
        s_het = torch.sigmoid(s_het)
        s_hom = torch.sigmoid(s_hom)
    s_het = torch.clamp(s_het, EPS, 1-EPS)
    s_hom = torch.clamp(s_hom, EPS, 1-EPS)

    lh = likelihood(s_het, s_hom, these_likelihoods)

    return (prior_p+lh, prior_p)


def likelihood(s_het, s_hom, likelihoods):
    '''
    p(y|parameter)
    '''
    S_GRID = torch.tensor(
        [0.] + np.exp(np.linspace(np.log(1e-5), 0, num=20)).tolist(),
        device=DEVICE
    )
    likelihoods -= torch.max(
        torch.max(likelihoods, dim=2, keepdim=True)[0],
        dim=1, keepdim=True
    )[0]

    with torch.no_grad():
        s_het_left_idx = torch.searchsorted(S_GRID, s_het, right=True) - 1
        s_hom_left_idx = torch.searchsorted(S_GRID, s_hom, right=True) - 1

    left_s_het = S_GRID[s_het_left_idx]
    right_s_het = S_GRID[s_het_left_idx + 1]
    left_s_hom = S_GRID[s_hom_left_idx]
    right_s_hom = S_GRID[s_hom_left_idx + 1]

    idx0 = torch.arange(s_het.shape[1])
    bottom_left_likelihood = []
    bottom_right_likelihood = []
    top_left_likelihood = []
    top_right_likelihood = []

    for i in range(likelihoods.shape[0]):
        likelihoods_curr = likelihoods[i].expand(s_het.shape[1], -1)
        bottom_left_likelihood.append(
            likelihoods_curr[
                idx0, s_het_left_idx[i, :], s_hom_left_idx[i, :]
            ]
        )
        bottom_right_likelihood.append(
            likelihoods_curr[
                idx0, s_het_left_idx[i, :]+1, s_hom_left_idx[i, :]
            ]
        )
        top_left_likelihood.append(
            likelihoods_curr[
                idx0, s_het_left_idx[i, :], s_hom_left_idx[i, :]+1
            ]
        )
        top_right_likelihood.append(
            likelihoods_curr[
                idx0, s_het_left_idx[i, :]+1, s_hom_left_idx[i, :]+1
            ]
        )

    bottom_left_likelihood = torch.stack(bottom_left_likelihood)
    bottom_right_likelihood = torch.stack(bottom_right_likelihood)
    top_left_likelihood = torch.stack(top_left_likelihood)
    top_right_likelihood = torch.stack(top_right_likelihood)

    bottom_weight = (right_s_hom - s_hom) / (right_s_hom - left_s_hom)
    left_weight = (right_s_het - s_het) / (right_s_het - left_s_het)

    sp_bl = bottom_left_likelihood + torch.log(
        bottom_weight * left_weight
    )
    sp_br = bottom_right_likelihood + torch.log(
        bottom_weight * (1 - left_weight)
    )
    sp_tl = top_left_likelihood + torch.log(
        (1-bottom_weight) * left_weight
    )
    sp_tr = top_right_likelihood + torch.log(
        (1-bottom_weight) * (1-left_weight)
     )
    success_p = (
        torch.exp(sp_bl)
        + torch.exp(sp_br)
        + torch.exp(sp_tl)
        + torch.exp(sp_tr)
    )

    return torch.log(torch.maximum(success_p, EPS))


class Prior(RegressionDistn):
    n_params = 5
    positive = np.array([False, False, True, True, False])
    scores = [PriorScore]

    def __init__(self, params):
        self._params = params
        self.params_transf = np.copy(params)
        self.params_transf[Prior.positive] = np.exp(
           self.params_transf[Prior.positive]
        )

    def distribution(params, logit=True):
        means = torch.tensor([params[0], params[1]]).T
        covs = torch.zeros((len(params[0]), 2, 2))
        covs[:, 0, 0] = params[2]
        covs[:, 1, 1] = params[3]
        corrs = 2*torch.sigmoid(params[4]) - 1
        covs[:, 0, 1] = torch.sqrt(params[2] * params[3]) * corrs
        covs[:, 1, 0] = covs[:, 0, 1]
        if logit:
            return dist.MultivariateNormal(means, covs)
        else:
            return dist.TransformedDistribution(
                dist.MultivariateNormal(means, covs),
                transforms=[dist.SigmoidTransform()])

    def get_mean(params):
        return params[0], params[1]

    def get_sd(params):
        return torch.sqrt(params[2]), torch.sqrt(params[3])

    def fit(y):
        """
        fit initial prior distribution for all genes
        """
        print("fitting initial distribution...")
        params = []
        for param in [0., 0., np.log(1.), np.log(1.), 0.]:
            params.append(
                torch.tensor(param, requires_grad=True, device=DEVICE)
            )
        optimizer = torch.optim.AdamW(params, lr=LR_INIT)

        for i in range(Prior.n_params):
            params[i] = params[i].expand(len(y), 1)

        lr_stage = 0
        min_loss = float('inf')

        for i in range(MAX_EPOCHS_INIT):
            loss = 0

            for idx in torch.split(
                torch.randperm(y.shape[0]), split_size_or_sections=BATCH_SIZE
            ):
                optimizer.zero_grad()
                these_params = [torch.exp(p[idx]) if Prior.positive[p_idx]
                                else p[idx] for p_idx, p in enumerate(params)]
                result = integrate(these_params, y[idx])
                result = -torch.sum(
                    torch.log(result)+torch.max(y[idx], dim=1)[0]
                )

                result.backward()
                optimizer.step()
                loss += result.item()

            print(
                "loss",
                loss,
                "params",
                params[0][0].item(),
                params[1][0].item(),
                params[2][0].item(),
                params[3][0].item(),
                params[4][0].item(),
                "lr",
                optimizer.param_groups[0]['lr']
            )

            if i == 0 or loss < min_loss:
                min_loss = loss
                min_epoch = i
            if i - min_epoch >= PATIENCE_INIT:
                optimizer.param_groups[0]['lr'] = (
                    optimizer.param_groups[0]['lr'] / 10
                )
                lr_stage += 1

            if lr_stage > 1:
                break

        params = [p[0].item() for p in params]
        print("initial params", params)
        params = np.array(params)

        return params

    @property
    def params(self):
        return self.params_transf


def compute_posterior(these_params, y, prob_y):
    grid = torch.linspace(-1., 1., N_INTEGRATION_PTS).expand(1, -1)
    means_s_het, means_s_hom = Prior.get_mean(these_params)
    sds_s_het, sds_s_hom = Prior.get_sd(these_params)
    grid_s_het = (
        means_s_het.expand(-1, 1) + grid * 8 * sds_s_het.expand(-1, 1)
    )
    grid_s_hom = (
        means_s_hom.expand(-1, 1) + grid * 8 * sds_s_hom.expand(-1, 1)
    )

    tile_grid_s_het = grid_s_het.repeat_interleave(len(grid_s_hom))
    tile_grid_s_hom = grid_s_hom.repeat(len(grid_s_het))

    eval_at_pts, prior_p = log_prob(Prior.distribution,
                                    these_params,
                                    tile_grid_s_het,
                                    tile_grid_s_hom,
                                    grid)
    eval_at_pts = eval_at_pts.reshape(-1, len(grid_s_het), len(grid_s_hom))

    eval_at_pts -= torch.log(prob_y)

    m = torch.max(
        torch.max(eval_at_pts, dim=1, keepdim=True)[0],
        dim=2,
        keepdim=True
    )[0]
    pdf = eval_at_pts - m

    # integral of g(x)f(x), where g = sigmoid
    pm_het = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*torch.sigmoid(grid_s_het.expand(1, -1, 1)),
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    pm_hom = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*torch.sigmoid(grid_s_hom.expand(1, 1, -1)),
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    h_grid = (
        torch.sigmoid(grid_s_het.expand(1, -1, 1))
        / torch.sigmoid(grid_s_hom.expand(1, 1, -1))
    )
    pm_h = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*h_grid,
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    sec_het = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*torch.sigmoid(grid_s_het.expand(1, -1, 1))**2,
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    sec_hom = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*torch.sigmoid(grid_s_hom.expand(1, 1, -1))**2,
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    sec_h = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(pdf)*h_grid**2,
            grid_s_hom,
            dim=2
        ),
        grid_s_het,
        dim=1
    )
    pv_het = sec_het - pm_het**2
    pv_hom = sec_hom - pm_hom**2
    pv_h = sec_h - pm_h**2

    m = torch.max(
        torch.max(prior_p, dim=1, keepdim=True)[0],
        dim=2,
        keepdim=True
    )[0]
    check = torch.exp(m.squeeze())*torch.trapezoid(
        torch.trapezoid(
            torch.exp(prior_p - m), grid_s_hom, dim=2
        ),
        grid_s_het,
        dim=1
    )
    if np.any(np.abs(check.detach().cpu().numpy() - 1.) > 1e-5):
        bad_idx = np.argmax(np.abs(check.detach().cpu().numpy()-1.))
        print('Discrepancy of ',
              np.max(np.abs(check.detach().cpu().numpy() - 1.)),
              'found while integrating')
        print('params were', [p[bad_idx] for p in these_params])
        print(torch.exp(prior_p[bad_idx, 0]), torch.exp(prior_p[bad_idx, -1]))
        print('mean was', means_s_het[bad_idx], means_s_hom[bad_idx])
        print('sd was', sds_s_het[bad_idx], sds_s_hom[bad_idx])
        print('grid was', grid[bad_idx])

    return pm_het, pm_hom, pm_h, pv_het, pv_hom, pv_h


def output_prior_posterior(model, feature_table, y, max_iter=None):
    print("calculating posterior distributions...")

    X = feature_table.to_numpy()
    params = torch.tensor(model.pred_dist(X, max_iter=max_iter)[:].params,
                          device=DEVICE)

    # prior
    prior_samples = Prior.distribution(params, logit=False).sample(
        torch.Size([10000])
    )
    prior_mean_s_het, prior_mean_s_hom = torch.mean(
        prior_samples,
        dim=0
    )
    prior_mean_h = torch.mean(prior_samples[:, 0] / prior_samples[:, 1])
    prior_mean_s_het, prior_mean_s_hom = torch.mean(
        Prior.distribution(params, logit=False).sample(torch.Size([10000])),
        dim=0
    ).detach().cpu().numpy()

    # posterior
    post_mean_s_het = []
    post_mean_s_hom = []
    post_mean_h = []
    post_var_s_het = []
    post_var_s_hom = []
    post_var_h = []

    for idx in range(params.shape[1]):
        idx = torch.tensor([idx], device=DEVICE)

        # compute p(y)
        prob_y = integrate(params[:, idx].unsqueeze(dim=-1), y[idx])

        # compute posterior pdf + expected posterior gene_property
        pm_het, pm_hom, pm_h, pv_het, pv_hom, pv_h = compute_posterior(
            params[:, idx].unsqueeze(dim=-1), y[idx], prob_y
        )

        post_mean_s_het.append(pm_het.item())
        post_mean_s_hom.append(pm_hom.item())
        post_mean_h.append(pm_h.item())
        post_var_s_het.append(pv_het.item())
        post_var_s_hom.append(pv_hom.item())
        post_var_h.append(pv_h.item())

    params = params.detach().cpu().numpy()
    output = pd.DataFrame({
        "ensg": feature_table.index,
        "param0": params[0],
        "param1": params[1],
        "param2": params[2],
        "param3": params[3],
        "param4": params[4],
        "prior_mean_s_het": prior_mean_s_het,
        "prior_mean_s_hom": prior_mean_s_hom,
        "prior_mean_h": prior_mean_h,
        "post_mean_s_het": post_mean_s_het,
        "post_mean_s_hom": post_mean_s_hom,
        "post_mean_h": post_mean_h,
        "post_var_s_het": post_var_s_het,
        "post_var_s_hom": post_var_s_hom,
        "post_var_h": post_var_h
    })
    output.to_csv(
        args.out_prefix + ".per_gene_estimates.tsv", sep='\t', index=None
    )

    # feature importance metrics
    importance = {"feature": feature_table.columns}
    for i in range(model.feature_importances_.shape[0]):
        importance["param%s_importance" % i] = model.feature_importances_[i]
    importance = pd.DataFrame(importance)
    importance.to_csv(
        args.out_prefix + ".feature_importance.tsv",
        sep='\t',
        index=False
    )


def format_lh(GENE_LIKELIHOODS, data, args):
    lh = []
    for gene in data.index:
        if gene not in GENE_LIKELIHOODS.keys():
            lh.append(torch.zeros([21, 21], device=DEVICE))
        else:
            lh_gene = (
                GENE_LIKELIHOODS[gene]['stop_gained'][args.sg][:, :]
                + GENE_LIKELIHOODS[gene]['splice_donor_variant'][args.sd][:, :]
                + GENE_LIKELIHOODS[gene]['splice_acceptor_variant'][args.sa][
                    :, :
                ]
            )
            lh.append(lh_gene.to(DEVICE))
    lh = torch.stack(lh)
    return lh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train_genes", dest="train_genes", required=False)
    parser.add_argument("--val_genes", dest="val_genes", required=False)
    parser.add_argument(
        "--gene_column",
        dest="gene_column",
        required=False,
        default="ensg",
        help="Name of the column containing gene names/IDs.",
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1, required=False
    )
    parser.add_argument(
        "--n_integration_pts",
        dest="n_integration_pts",
        type=int,
        default=1001,
        required=False,
        help="Number of points for numerical integration. Larger values "
             "increase can improve accuracy but also increase training "
             "time and/or numerical instability.",
    )
    parser.add_argument(
        "--total_iterations",
        dest="total_iterations",
        type=int,
        default=500,
        required=False,
        help="Maximum number of iterations. The actual number of iterations "
             "may be lower if using early stopping (see the 'train' option)",
    )
    parser.add_argument(
        "--early_stopping_iter",
        dest="early_stopping_iter",
        type=int,
        default=10,
        required=False,
        help="If >0, chromosomes 2, 4, 6 will be held out for validation "
             "and training will end when the loss on these chromosomes stops "
             "decreasing for the specified number of iterations. Otherwise, "
             "the model will train on all genes for the number of iterations "
             "specified in 'total_iterations'.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.05,
        required=False,
        help="Learning rate for NGBoost. Smaller values may improve accuracy "
             "but also increase training time. Typical values: 0.01 to 0.2.",
    )
    parser.add_argument(
        "--max_depth",
        dest="max_depth",
        type=int,
        default=3,
        required=False,
        help="XGBoost parameter. See https://xgboost.readthedocs.io/"
             "en/stable/python/python_api.html.",
    )
    parser.add_argument(
        "--n_trees_per_iteration",
        dest="n_estimators",
        type=int,
        default=1,
        required=False,
        help="XGBoost parameter, n_estimators",
    )
    parser.add_argument(
        "--min_child_weight",
        dest="min_child_weight",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--reg_alpha",
        dest="reg_alpha",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--reg_lambda",
        dest="reg_lambda",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--subsample",
        dest="subsample",
        type=float,
        required=False,
        help="XGBoost parameter.",
    )
    parser.add_argument(
        "--sg_err",
        dest="sg",
        type=int,
        default=0,
        help="Index for the error rate to use for stop gain variants.",
    )
    parser.add_argument(
        "--sd_err",
        dest="sd",
        type=int,
        default=0,
        help="Index for the error rate to use for splice donor variants.",
    )
    parser.add_argument(
        "--sa_err",
        dest="sa",
        type=int,
        default=0,
        help="Index for the error rate to use splice acceptor variants.",
    )
    parser.add_argument(
        "--hess_reg_1",
        dest="hess_reg_1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hess_reg_2",
        dest="hess_reg_2",
        type=float,
        default=0.5,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--response",
        dest="response",
        required=True,
        help="tsv containing data that can be related to the gene property of "
             "interest through a likelihood function. Also known as y or "
             "dependent variable.",
    )
    required.add_argument(
        "--features",
        dest="features",
        required=True,
        help="tsv containing gene features. Also known as X or independent "
             "variables.",
    )
    required.add_argument(
        "--out", dest="out_prefix", help="Prefix for the output files."
    )
    required.add_argument(
        "--integration_lb",
        dest="integration_lb",
        type=float,
        help="Lower bound for numerical integration - the smallest value that "
             "you expect for the gene property of interest.",
    )
    required.add_argument(
        "--integration_ub",
        dest="integration_ub",
        type=float,
        help="Upper bound for numerical integration - the largest value you "
             "expect for the gene property of interest.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Provide a pretrained model to obtain predictions for "
             "that model."
    )

    args = parser.parse_args()
    print(vars(args))

    global LR_INIT, N_METRIC, MAX_EPOCHS_INIT, PATIENCE_INIT
    global DEVICE, ZERO
    global HESS_REG_1, HESS_REG_2

    MAX_EPOCHS_INIT = 2
    LR_INIT = 1e-3
    PATIENCE_INIT = 1
    N_METRIC = 1000
    N_INTEGRATION_PTS = args.n_integration_pts
    GENE_COLUMN = args.gene_column
    BATCH_SIZE = args.batch_size
    HESS_REG_1 = args.hess_reg_1
    HESS_REG_2 = args.hess_reg_2

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEVICE == torch.device("cuda:0"):
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    ZERO = torch.tensor(0., device=DEVICE)
    EPS = torch.tensor(1e-15, device=DEVICE)

    # load likelihoods and training data
    y = pickle.load(open(args.response, 'rb'))
    feature_table = pd.read_csv(args.features, sep='\t', index_col="ensg")
    all_genes = list(set(feature_table.index))
    feature_table = feature_table.loc[all_genes]
    y = format_lh(y, feature_table, args)

    # train
    train_idx = ~feature_table["chrom"].isin(
        ["chr1", "chr3", "chr5", "chr2", "chr4", "chr6"]
    )
    val_idx = feature_table["chrom"].isin(["chr2", "chr4", "chr6"])

    feature_table = feature_table.drop(["hgnc", "chrom"], axis=1)
    X = feature_table.to_numpy()

    X_train, y_train = X[train_idx], y[train_idx.to_numpy()]
    X_val, y_val = X[val_idx], y[val_idx.to_numpy()]

    # already have pretrained model, want to obtain predictions
    if args.model is not None:
        model = pickle.load(open(args.model, 'rb'))
    else:  # train model and make predictions
        xgb_params = {"max_depth": args.max_depth,
                      "reg_alpha": args.reg_alpha,
                      "reg_lambda": args.reg_lambda,
                      "min_child_weight": args.min_child_weight,
                      "eta": 1.0,
                      "subsample": args.subsample,
                      "n_estimators": args.n_estimators}
        xgb_params = {k: v for k, v in xgb_params.items() if v is not None}

        if not torch.cuda.is_available():
            learner = xgb.XGBRegressor(
                tree_method="hist",
                **xgb_params
            )
        else:
            learner = xgb.XGBRegressor(
                device=DEVICE,
                tree_method="hist",
                **xgb_params
            )

        print("X.shape", X.shape)
        print("X_train.shape", X_train.shape)
        print("X_val.shape", X_val.shape)

        if args.early_stopping_iter > 0:
            model = NGBRegressor(
                n_estimators=args.total_iterations,
                Dist=Prior,
                Base=learner,
                Score=PriorScore,
                verbose_eval=1,
                learning_rate=args.lr,
                natural_gradient=False
            ).fit(
                X_train,
                y_train,
                X_val=X_val,
                Y_val=y_val,
                early_stopping_rounds=args.early_stopping_iter
            )
        else:
            print("no validation set")
            model = NGBRegressor(
                n_estimators=args.total_iterations,
                Dist=Prior,
                Base=learner,
                Score=PriorScore,
                verbose_eval=1,
                learning_rate=args.lr,
                natural_gradient=False
            ).fit(X, y)

        f = open(args.out_prefix + ".model", 'wb')
        pickle.dump(model, f)
        f.close()

    if model.best_val_loss_itr is not None:
        output_prior_posterior(
            model, feature_table, y, model.best_val_loss_itr
        )
    else:
        output_prior_posterior(model, feature_table, y)
