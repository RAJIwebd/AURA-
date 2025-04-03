package com.example.blind_as

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.scale
import java.nio.FloatBuffer

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
        val floatBuffer = FloatBuffer.wrap(inputTensor)
        val inputShape = longArrayOf(1, 3, 256, 256)

        val inputs = mapOf(inputName to OnnxTensor.createTensor(ortEnvironment, floatBuffer, inputShape))
        val output = session.run(inputs)
        val outputTensor = output[0] as OnnxTensor

        val outputArray = outputTensor.floatBuffer.array()
        val numBoxes = 1344

        val detectedRegions = mutableListOf<Region>()
        val detectedCategories = mutableListOf<Pair<String, Float>>()

        // Category labels from your Python script
        val categories = arrayOf(
            "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
            "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED",
            "FACE_MALE", "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED",
            "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"
        )

        Log.d("ONNXModelHelper", "Parsing Model Output... Found $numBoxes boxes.")

        // Process detected objects
        for (i in 0 until numBoxes) {
            val offset = i * 22  // Each box has 22 attributes
            val x1 = outputArray[offset]
            val y1 = outputArray[offset + 1]
            val x2 = outputArray[offset + 2]
            val y2 = outputArray[offset + 3]
            val score = outputArray[offset + 4]

            // Extract category probabilities (offset + 5 to offset + 22)
            val scores = outputArray.sliceArray((offset + 5) until (offset + 22))
            val maxScore = scores.maxOrNull() ?: 0f
            val detectedCategoryIndex = scores.toList().indexOf(maxScore)

            if (score > 0.5f && maxScore > 0.1f) {
                val detectedCategory = categories[detectedCategoryIndex]
                detectedRegions.add(Region(x1, y1, x2, y2, score))
                Log.d("ONNXModelHelper", "🔴 Detected: $detectedCategory (Confidence: ${"%.2f".format(maxScore)})")
            }
        }


        outputTensor.close()

        // Log all detected categories
        if (detectedCategories.isNotEmpty()) {
            Log.d("ONNXModelHelper", "🚨 Detected Sensitive Areas: $detectedCategories")
        } else {
            Log.d("ONNXModelHelper", "✅ No explicit content detected.")
        }

        return detectedRegions
    }




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

            floatValues[i] = b  // Swap R and B if needed
            floatValues[i + width * height] = g
            floatValues[i + 2 * width * height] = r

        }

        Log.d("ONNXModelHelper", "Image Preprocessing Completed!")
        return floatValues
    }



    fun pixelateRegions(bitmap: Bitmap, regions: List<Region>): Bitmap {
        Log.d("ONNXModelHelper", "Starting Pixelation Process...")

        val pixelSize = 15
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.style = Paint.Style.FILL

        val width = bitmap.width
        val height = bitmap.height

        for (region in regions) {
            val x1 = (region.x1 * width).toInt()
            val y1 = (region.y1 * height).toInt()
            val x2 = (region.x2 * width).toInt()
            val y2 = (region.y2 * height).toInt()

            Log.d("ONNXModelHelper", "Pixelating Region: ($x1, $y1) -> ($x2, $y2)")

            for (y in y1 until y2 step pixelSize) {
                for (x in x1 until x2 step pixelSize) {
                    if (x < width && y < height) {
                        val avgColor = mutableBitmap[x, y]
                        paint.color = avgColor
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

        // Call savePixelatedImage function from CameraActivity



        return mutableBitmap

        Log.d("ONNXModelHelper", "Pixelation Completed Successfully!")
        return mutableBitmap
    }



    data class Region(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)




    fun closeSession() {
        Log.d("ONNXModelHelper", "Closing ONNX Model Session...")
        session.close()
        ortEnvironment.close()
        Log.d("ONNXModelHelper", "ONNX Model Session Closed Successfully!")
    }
}