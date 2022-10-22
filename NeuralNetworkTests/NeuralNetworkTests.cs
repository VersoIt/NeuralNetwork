using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;
using PictureConverter;
using NeuralNetwork;
using Microsoft.VisualStudio.TestPlatform.Common.DataCollection;
using NeuralNetwork.Tools;

namespace NeuronTests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        const int Width = 20;
        const int Height = 20;

        [TestMethod()]
        public void MainTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //T  A  S  F
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };

            var topology = new NeuralNetworkTopology(4, 1, 0.1, 2);
            var neuralNetwork = new FeedForwardNeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 10000);

            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = Matrix.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod()]
        public void FeedForwardTest()
        {
            var parasitizedPath = @"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\Parazitized\";
            var unparasitizedPath = @"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\Parazitized\";

            var converter = new ImageBinarizer();

            var testParasitizedImageInput = converter.ConvertIntoBinarizedArray(@"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\Parazitized\1.png");
            var testUnparasitizedImageInput = converter.ConvertIntoBinarizedArray(@"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\Unparazitized\1.png");

            var topology = new NeuralNetworkTopology(400, 1, 0.1, 200);
            var neuralNetwork = new FeedForwardNeuralNetwork(topology);

            converter.Convert(@"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\Parazitized\1.png").Save(@"C:\Users\Ruslan\source\repos\NeuralNetwork\NeuralNetworkTests\tttt.png");

            double[,] parasitizedInputs = GetData(parasitizedPath, converter, 100);
            neuralNetwork.Learn(new double[] { 1 }, parasitizedInputs, 1);

            double[,] unparasitizedInputs = GetData(unparasitizedPath, converter, 100);
            neuralNetwork.Learn(new double[] { 0 }, unparasitizedInputs, 100);

            var par = neuralNetwork.FeedForward(testParasitizedImageInput);
            var unpar = neuralNetwork.FeedForward(testUnparasitizedImageInput);

            Assert.AreEqual(1, Math.Round(par.Output, 2));
            Assert.AreEqual(0, Math.Round(unpar.Output, 2));
        }

        private static double[,] GetData(string parazitizedPath, ImageBinarizer converter, int size)
        {
            var images = Directory.GetFiles(parazitizedPath);
            var result = new double[size, Width * Height];

            for (int i = 0; i < size; ++i)
            {
                var image = converter.ConvertIntoBinarizedArray(images[i]);
                for (int j = 0; j < Width * Height; ++j)
                {
                    result[i, j] = image[j];
                }
            }

            return result;
        }
    }
}