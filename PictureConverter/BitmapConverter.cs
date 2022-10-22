using System.Drawing;
using System.IO;
using static System.Net.Mime.MediaTypeNames;

namespace PictureConverter
{
    public class ImageBinarizer : Abstract.IConvertible<Bitmap, Bitmap>, Abstract.IConvertible<Bitmap, string>
    {
        public const int Boundary = 128;

        public Bitmap Convert(Bitmap image)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);
            for (int x = 0; x < image.Width; ++x)
            {
                for (int y = 0; y < image.Height; ++y)
                {
                    result.SetPixel(x, y, (GetBrightnessPixelValue(image.GetPixel(x, y)) == 0 ? Color.White : Color.Black));
                }
            }
            return result;
        }

        public double[] ConvertIntoBinarizedArray(Bitmap image)
        {
            var result = new double[image.Width * image.Height];
            for (int x = 0; x < image.Width; ++x)
            {
                for (int y = 0; y < image.Height; ++y)
                {
                    result[y * image.Width + x] = GetBrightnessPixelValue(image.GetPixel(x, y));
                }
            }
            return result;
        }

        public Bitmap Convert(string path)
        {
            var result = new Bitmap(new Bitmap(path), new Size(20, 20));
            for (int x = 0; x < 20; ++x)
            {
                for (int y = 0; y < 20; ++y)
                {
                    result.SetPixel(x, y, (GetBrightnessPixelValue(result.GetPixel(x, y)) == 0 ? Color.White : Color.Black));
                }
            }
            return result;
        }

        public double[] ConvertIntoBinarizedArray(string path)
        {
            var resizedImage = new Bitmap(new Bitmap(path), new Size(20, 20));
            var result = new double[resizedImage.Width * resizedImage.Height];

            for (int x = 0; x < resizedImage.Width; ++x)
            {
                for (int y = 0; y < resizedImage.Height; ++y)
                {
                    result[y * resizedImage.Width + x] = GetBrightnessPixelValue(resizedImage.GetPixel(x, y));
                }
            }
            return result;
        }

        public Bitmap Test(double[] pixels, int a = 20, int b = 20)
        {
            var result = new Bitmap(a, b);

            for (int x = 0; x < 20; ++x)
            {
                for (int y = 0; y < 20; ++y)
                {

                    result.SetPixel(x, y, pixels[y * b + x] == 0 ? Color.White : Color.Black);
                }
            }

            return result;
        }

        private int GetBrightnessPixelValue(Color pixel)
        {
            double transition = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
            return transition < Boundary ? 0 : 1;
        }
    }
}