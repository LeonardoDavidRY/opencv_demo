#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
 
using namespace cv;
using namespace std;
//g++ detectar_rostros.cpp -o detectar_rostros `pkg-config --cflags --libs opencv4`
 
int main() {
    string imagen_path = "personas2.jpg";  // cambia si tu imagen tiene otro nombre
 
    // Verificar que exista la imagen
    if (!std::filesystem::exists(imagen_path)) {
        cerr << "❌ No se encontró la imagen '" << imagen_path << "' en la carpeta actual." << endl;
        return -1;
    }
 
    // Cargar el clasificador de rostros
    string cascade_path = samples::findFile("haarcascades/haarcascade_frontalface_default.xml");
    CascadeClassifier face_cascade;
 
    if (!face_cascade.load(cascade_path)) {
        cerr << "❌ Error al cargar el clasificador de rostros." << endl;
        return -1;
    }
 
    // Leer la imagen
    Mat imagen = imread(imagen_path);
    if (imagen.empty()) {
        cerr << "❌ No se pudo cargar la imagen. Revisa el formato o la ruta." << endl;
        return -1;
    }
 
    // Convertir a escala de grises
    Mat gray;
    cvtColor(imagen, gray, COLOR_BGR2GRAY);
 
    // Detectar rostros
    vector<Rect> rostros;
    face_cascade.detectMultiScale(
        gray,
        rostros,
        1.1,  // scaleFactor
        4,    // minNeighbors
        0,    // flags
        Size(30, 30) // tamaño mínimo
    );
 
    // Dibujar rectángulos alrededor de los rostros detectados
    for (const Rect& rostro : rostros) {
        rectangle(imagen, rostro, Scalar(0, 255, 0), 2);
    }
 
    // Guardar la imagen resultante
    string resultado_path = "resultado.jpg";
    imwrite(resultado_path, imagen);
 
    cout << "✅ Se detectaron " << rostros.size() << " rostro(s). Imagen guardada como '" << resultado_path << "'." << endl;
    return 0;
}
 