using Distributions
import Distributions: length, insupport, _logpdf
export PDFDistribution

type PDFDistribution <: ContinuousUnivariateDistribution
  pdf::Function
end

function _logpdf(d::PDFDistribution, x)
  log(d.pdf(x))
end
