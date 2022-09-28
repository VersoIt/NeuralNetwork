
namespace Neuron.Tools
{
    internal class SigmoidDx : MathFunction
    {
        public override double GetResultBy(double x)
        {
            var sigmoid = (new Sigmoid()).GetResultBy(x);
            return sigmoid / (1 - sigmoid);
        }
    }
}
