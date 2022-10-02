using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;

namespace PictureConverter.Tests
{
    [TestClass()]
    public class PictureConverterTests
    {
        [TestMethod()]
        public void ConvertTest()
        {
            var converter = new ImageBinarizer(new Bitmap(@"C:\Users\Ruslan\source\repos\Neuron\123123.jpg"));
            var result = converter.Convert();
            result.Save(@"C:\Users\Ruslan\source\repos\Neuron\1231.png");
        }
    }
}