namespace Neuron.Tools
{
    internal class Sigmoid : MathFunction
    {
        public override double GetResultBy(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
    }
}
