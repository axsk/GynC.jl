
" generate a gync likelihoodmodel "
function gyncmodel(xs, datas; zmult = 0, sigma=0.1)
  phi(x) = GynC.forwardsol(x)[:,GynC.measuredinds]
  ys = phi.(xs);

  nonaninds = find(x->!any(isnan(x)), ys)

  xs = xs[nonaninds]
  ys = ys[nonaninds]

  err = GynC.MatrixNormalCentered(repmat(sigma*GynC.model_measerrors' * 10, 31)) # TODO: 10 hotfix for static scaling in model.jl

  zs = map(y->y+rand(err), repmat(ys, zmult));

  m = GynC.LikelihoodModel(xs, ys, zs, datas, err);
end
