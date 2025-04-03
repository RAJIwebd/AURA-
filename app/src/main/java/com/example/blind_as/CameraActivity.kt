package com.example.blind_as

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors





class CameraActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageCapture: ImageCapture
    private lateinit var previewView: PreviewView
    private lateinit var captureButton: Button
    private lateinit var uploadButton: Button
    private lateinit var imageView: ImageView
    private lateinit var onnxHelper: ONNXModelHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        captureButton = findViewById(R.id.captureButton)
        uploadButton = findViewById(R.id.uploadButton)
        imageView = findViewById(R.id.imageView)
        onnxHelper = ONNXModelHelper(this)
        cameraExecutor = Executors.newSingleThreadExecutor()



        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        captureButton.setOnClickListener {
            captureImage()
        }

        uploadButton.setOnClickListener {
            openGallery()
        }
    }

    /**
     * ✅ Checks if the required permissions are granted
     */
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    /**
     * ✅ Initializes and starts the camera using CameraX
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            imageCapture = ImageCapture.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CameraActivity", "Camera start failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * ✅ Opens gallery to select an image
     */
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val inputStream: InputStream? = contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                processImage(bitmap)
            } catch (e: Exception) {
                Log.e("CameraActivity", "Error reading image", e)
            }
        }
    }

    private fun openGallery() {
        galleryLauncher.launch("image/*")
    }

    /**
     * ✅ Processes the selected image with ONNX model
     */
    private fun processImage(bitmap: Bitmap) {
        Log.d("CameraActivity", "📸 Image uploaded successfully, processing...")

        val regions = onnxHelper.runInference(bitmap)

        if (regions.isNotEmpty()) {
            val pixelatedBitmap = onnxHelper.pixelateRegions(bitmap, regions)


            // Save the pixelated image

            savePixelatedImage(pixelatedBitmap, this)


            runOnUiThread {
                imageView.setImageBitmap(pixelatedBitmap)
            }
            Log.d("CameraActivity", "🚩 Explicit content detected, regions: $regions")
        } else {
            runOnUiThread {
                imageView.setImageBitmap(bitmap)
            }
            Log.d("CameraActivity", "✅ No explicit content detected.")
        }
    }

    /**
     * ✅ Captures an image using CameraX and processes it
     */
    private fun captureImage() {
        val photoFile = File(filesDir, "image.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    val regions = onnxHelper.runInference(bitmap)

                    runOnUiThread {
                        if (regions.isNotEmpty()) {
                            val pixelatedBitmap = onnxHelper.pixelateRegions(bitmap, regions)
                            imageView.setImageBitmap(pixelatedBitmap)
                            savePixelatedImage(pixelatedBitmap, this@CameraActivity)
                        } else {
                            imageView.setImageBitmap(bitmap)
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("ONNX_ERROR", "Image capture failed", exception)
                }
            }
        )
    }

    /**
     * ✅ Saves the pixelated image and updates the gallery
     */
    private fun savePixelatedImage(bitmap: Bitmap, context: Context) {
        try {
            // Define the folder in Pictures
            val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val appDir = File(picturesDir, "BlindAS") // Folder name: "BlindAS"
            if (!appDir.exists()) appDir.mkdirs() // Create folder if it doesn't exist

            // Generate a unique filename to avoid overwriting
            val photoFile = File(appDir, "pixelated_${System.currentTimeMillis()}.jpg")

            // Save the image
            FileOutputStream(photoFile).use { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            }

            // Notify the media scanner to update the Gallery
            MediaScannerConnection.scanFile(context, arrayOf(photoFile.absolutePath), null) { _, uri ->
                Log.d("CameraActivity", "✅ Image saved and updated in gallery: $uri")
            }

            Log.d("CameraActivity", "✅ Image successfully saved at: ${photoFile.absolutePath}")
        } catch (e: Exception) {
            Log.e("CameraActivity", "❌ Failed to save image: ${e.message}")
        }
    }


    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
