package com.example.blind_as

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import ai.onnxruntime.*
import android.util.Log
import java.nio.FloatBuffer
import androidx.core.graphics.scale
import androidx.core.graphics.get

class ONNXModelHelper(context: Context) {

    private var ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession

    init {
        Log.d("ONNXModelHelper", "Initializing ONNX Model Helper...")
        val modelPath = context.assets.open("your_model_v9.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        session = ortEnvironment.createSession(modelPath, options)
        Log.d("ONNXModelHelper", "ONNX Model Loaded Successfully!")
    }

    fun runInference(bitmap: Bitmap): List<Region> {
        Log.d("ONNXModelHelper", "Running Inference on the Image...")
        val inputTensor = preprocessImage(bitmap)
        val inputName = session.inputNames.first()
        Log.d("ONNXModelHelper", "Input Tensor Name: $inputName")

        val floatBuffer = FloatBuffer.wrap(inputTensor)
        val inputShape = longArrayOf(1, 3, 256, 256)
        Log.d("ONNXModelHelper", "Input Shape: ${inputShape.contentToString()}")

        val inputs = mapOf(inputName to OnnxTensor.createTensor(ortEnvironment, floatBuffer, inputShape))
        val output = session.run(inputs)
        val outputTensor = output[0] as OnnxTensor

        Log.d("ONNXModelHelper", "Model Inference Completed!")

        val rawOutput = outputTensor.value as Array<Array<FloatArray>>
        val results = rawOutput[0]

        val detectedRegions = mutableListOf<Region>()

        Log.d("ONNXModelHelper", "Parsing Model Output...")

        for (result in results) {
            if (result.size == 5) {  // Assuming the model returns [x1, y1, x2, y2, score]
                val x1 = result[0]
                val y1 = result[1]
                val x2 = result[2]
                val y2 = result[3]
                val score = result[4]

                if (score > 0.5f) { // Confidence threshold
                    detectedRegions.add(Region(x1, y1, x2, y2, score))
                    Log.d("ONNXModelHelper", "Detected Region -> x1: $x1, y1: $y1, x2: $x2, y2: $y2, Score: $score")
                }
            }
        }

        outputTensor.close()
        Log.d("ONNXModelHelper", "Inference and Parsing Completed! Total Regions Detected: ${detectedRegions.size}")
        return detectedRegions
    }

    fun pixelateRegions(bitmap: Bitmap, regions: List<Region>): Bitmap {
        Log.d("ONNXModelHelper", "Starting Pixelation Process...")

        val pixelSize = 15
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()

        val width = bitmap.width
        val height = bitmap.height

        Log.d("ONNXModelHelper", "Image Width: $width, Height: $height")

        for (region in regions) {
            val x1 = (region.x1 * width).toInt()
            val y1 = (region.y1 * height).toInt()
            val x2 = (region.x2 * width).toInt()
            val y2 = (region.y2 * height).toInt()

            Log.d("ONNXModelHelper", "Pixelating Region: ($x1, $y1) -> ($x2, $y2)")

            for (y in y1 until y2 step pixelSize) {
                for (x in x1 until x2 step pixelSize) {
                    if (x < width && y < height) {
                        val pixelColor = mutableBitmap[x, y]
                        paint.color = pixelColor
                        canvas.drawRect(
                            x.toFloat(),
                            y.toFloat(),
                            (x + pixelSize).coerceAtMost(width).toFloat(),
                            (y + pixelSize).coerceAtMost(height).toFloat(),
                            paint
                        )
                    }
                }
            }
        }

        Log.d("ONNXModelHelper", "Pixelation Completed Successfully!")
        return mutableBitmap
    }

    data class Region(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

    fun preprocessImage(bitmap: Bitmap): FloatArray {
        Log.d("ONNXModelHelper", "Preprocessing Image for Inference...")
        val width = 256
        val height = 256
        val floatValues = FloatArray(3 * width * height)

        val resizedBitmap = bitmap.scale(width, height)
        val intValues = IntArray(width * height)
        resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        for (i in intValues.indices) {
            val pixel = intValues[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            floatValues[i] = r
            floatValues[i + width * height] = g
            floatValues[i + 2 * width * height] = b
        }

        Log.d("ONNXModelHelper", "Image Preprocessing Completed!")
        return floatValues
    }

    fun closeSession() {
        Log.d("ONNXModelHelper", "Closing ONNX Model Session...")
        session.close()
        ortEnvironment.close()
        Log.d("ONNXModelHelper", "ONNX Model Session Closed Successfully!")
    }
}
