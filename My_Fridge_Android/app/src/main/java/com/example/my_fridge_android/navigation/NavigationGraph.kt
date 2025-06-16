package com.example.my_fridge_android.navigation

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.my_fridge_android.navigation.Screen.Login
import com.example.my_fridge_android.navigation.Screen.Registe
import com.example.my_fridge_android.navigation.Screen.Home
import com.example.my_fridge_android.navigation.Screen.RecieptList
import com.example.my_fridge_android.navigation.Screen.Profile
import com.example.my_fridge_android.navigation.Screen.RecipeChat
import com.example.my_fridge_android.ui.login.LoginScreen
import com.example.my_fridge_android.ui.login.LoginViewModel
import com.example.my_fridge_android.ui.registe.RegisteScreen
import com.example.my_fridge_android.ui.registe.RegisteViewModel
import com.example.my_fridge_android.ui.home.HomeScreen
import com.example.my_fridge_android.ui.home.HomeViewModel
import com.example.my_fridge_android.ui.recieptlist.RecieptListScreen
import com.example.my_fridge_android.ui.recieptlist.RecieptListViewModel
import com.example.my_fridge_android.ui.profile.ProfileScreen
import com.example.my_fridge_android.ui.profile.ProfileViewModel
import com.example.my_fridge_android.ui.recipechat.RecipeChatScreen
import com.example.my_fridge_android.ui.recipechat.RecipeChatViewModel
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun NavigationGraph(
    navController: NavHostController,
    startDestination: Screen,
    modifier: Modifier = Modifier,
) {
    NavHost(
        modifier = modifier,
        navController = navController,
        startDestination = startDestination,
    ) {
        composable<Login>(
            enterTransition = { fadeIn(animationSpec = tween(300)) },
            exitTransition = { fadeOut(animationSpec = tween(300)) },
            popEnterTransition = { fadeIn(animationSpec = tween(300)) },
            popExitTransition = { fadeOut(animationSpec = tween(300)) }
        ) {
            val viewModel: LoginViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect
            LoginScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction,
                onNavigateToHome = {
                    navController.navigate(Home) {
                        popUpTo(Login) { inclusive = true }
                    }
                },
                onNavigateToSignUp = {
                    navController.navigate(Registe)
                },
                onNavigateToForgotPassword = {
                    // Navigate to forgot password screen when implemented
                }
            )
        }
        composable<Registe>(
            enterTransition = { slideInHorizontally(animationSpec = tween(300)) },
            exitTransition = { slideOutHorizontally(animationSpec = tween(300)) },
            popEnterTransition = { slideInHorizontally(animationSpec = tween(300)) },
            popExitTransition = { slideOutHorizontally(animationSpec = tween(300)) }
        ) {
            val viewModel: RegisteViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect
            RegisteScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction,
                onNavigateToLogin = {
                    navController.popBackStack()
                },
                onNavigateToHome = {
                    navController.navigate(Home) {
                        popUpTo(Login) { inclusive = true }
                    }
                }
            )
        }
        composable<Home>(
            enterTransition = { fadeIn(animationSpec = tween(300)) },
            exitTransition = { fadeOut(animationSpec = tween(300)) },
            popEnterTransition = { fadeIn(animationSpec = tween(300)) },
            popExitTransition = { fadeOut(animationSpec = tween(300)) }
        ) {
            val viewModel: HomeViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect

            // Handle navigation effects
            LaunchedEffect(uiEffect) {
                uiEffect.collect { effect ->
                    when (effect) {
                        is com.example.my_fridge_android.ui.home.HomeContract.UiEffect.NavigateToReceiptList -> {
                            navController.navigate(RecieptList)
                        }

                        is com.example.my_fridge_android.ui.home.HomeContract.UiEffect.NavigateToRecipeChat -> {
                            navController.navigate(RecipeChat)
                        }

                        is com.example.my_fridge_android.ui.home.HomeContract.UiEffect.NavigateToRecipeChatWithResponse -> {
                            // Set the API response in the RecipeChatViewModel
                            println("DEBUG: NavigationGraph - Setting API response: ${effect.response}")
                            com.example.my_fridge_android.ui.recipechat.RecipeChatViewModel.setApiResponse(
                                effect.response
                            )
                            navController.navigate(RecipeChat)
                        }

                        is com.example.my_fridge_android.ui.home.HomeContract.UiEffect.NavigateToLogin -> {
                            navController.navigate(Login) {
                                popUpTo(Home) { inclusive = true }
                            }
                        }

                        is com.example.my_fridge_android.ui.home.HomeContract.UiEffect.ShowMessage -> {
                            // Message is handled by the HomeScreen itself via snackbar
                        }
                    }
                }
            }

            HomeScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction
            )
        }
        composable<RecieptList>(
            enterTransition = { slideInHorizontally(animationSpec = tween(300)) },
            exitTransition = { slideOutHorizontally(animationSpec = tween(300)) },
            popEnterTransition = { slideInHorizontally(animationSpec = tween(300)) },
            popExitTransition = { slideOutHorizontally(animationSpec = tween(300)) }
        ) {
            val viewModel: RecieptListViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect
            val context = LocalContext.current

            // Load fridge items when entering the screen
            LaunchedEffect(Unit) {
                viewModel.onAction(com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiAction.LoadFridgeItems)
            }

            // Camera functionality
            var photoUri by remember { mutableStateOf<Uri?>(null) }
            var isCameraLaunching by remember { mutableStateOf(false) }

            val cameraLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.TakePicture()
            ) { success ->
                isCameraLaunching = false
                viewModel.resetCameraLaunching()
                try {
                    if (success) {
                        photoUri?.let { uri ->
                            // Photo was taken successfully
                            println("Photo taken successfully: $uri")
                            // Upload the image to the Flask server
                            viewModel.uploadImage(uri)
                        }
                    } else {
                        // Photo capture cancelled or failed
                        println("Photo capture cancelled or failed")
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    println("Error processing photo result: ${e.message}")
                } finally {
                    photoUri = null
                }
            }

            // Permission launchers
            val cameraPermissionLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.RequestPermission()
            ) { isGranted ->
                if (isGranted) {
                    try {
                        if (isCameraLaunching) {
                            println("Camera already launching, ignoring duplicate request")
                            return@rememberLauncherForActivityResult
                        }

                        isCameraLaunching = true
                        // Permission granted, launch camera
                        val photoFile = File(
                            context.getExternalFilesDir(null),
                            "receipt_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.jpg"
                        )

                        // Ensure the parent directory exists
                        photoFile.parentFile?.mkdirs()

                        val uri = FileProvider.getUriForFile(
                            context,
                            "${context.packageName}.provider",
                            photoFile
                        )
                        photoUri = uri

                        // Use Handler for delay instead of coroutines
                        val handler = android.os.Handler(context.mainLooper)
                        handler.postDelayed({
                            try {
                                cameraLauncher.launch(uri)
                            } catch (e: Exception) {
                                isCameraLaunching = false
                                viewModel.resetCameraLaunching()
                                e.printStackTrace()
                                println("Failed to launch camera: ${e.message}")
                            }
                        }, 200)
                    } catch (e: Exception) {
                        // Handle any errors gracefully
                        isCameraLaunching = false
                        viewModel.resetCameraLaunching()
                        e.printStackTrace()
                        println("Error setting up camera: ${e.message}")
                    }
                } else {
                    isCameraLaunching = false
                    viewModel.resetCameraLaunching()
                    println("Camera permission denied")
                }
            }

            val galleryLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.GetContent()
            ) { uri ->
                if (uri != null) {
                    // Image was selected from gallery
                    println("Image selected from gallery: $uri")
                    // Upload the selected image to the Flask server
                    viewModel.uploadImage(uri)
                }
            }

            // Handle navigation effects
            LaunchedEffect(uiEffect) {
                uiEffect.collect { effect ->
                    when (effect) {
                        is com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect.NavigateToHome -> {
                            navController.popBackStack()
                        }
                        is com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect.OpenCamera -> {
                            // Request camera permission first
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                        is com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect.OpenGallery -> {
                            galleryLauncher.launch("image/*")
                        }
                        is com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect.ShowError -> {
                            // Handle error display - could show a toast or snackbar
                            println("Error: ${effect.message}")
                        }
                        is com.example.my_fridge_android.ui.recieptlist.RecieptListContract.UiEffect.ShowMessage -> {
                            // Handle success message display - handled by RecieptListScreen
                            println("Message: ${effect.message}")
                        }
                    }
                }
            }

            RecieptListScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction
            )
        }
        composable<Profile>(
            enterTransition = { fadeIn(animationSpec = tween(300)) },
            exitTransition = { fadeOut(animationSpec = tween(300)) },
            popEnterTransition = { fadeIn(animationSpec = tween(300)) },
            popExitTransition = { fadeOut(animationSpec = tween(300)) }
        ) {
            val viewModel: ProfileViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect
            ProfileScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction
            )
        }
        composable<RecipeChat>(
            enterTransition = { fadeIn(animationSpec = tween(300)) },
            exitTransition = { fadeOut(animationSpec = tween(300)) },
            popEnterTransition = { fadeIn(animationSpec = tween(300)) },
            popExitTransition = { fadeOut(animationSpec = tween(300)) }
        ) {
            val viewModel: RecipeChatViewModel = hiltViewModel()
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val uiEffect = viewModel.uiEffect
            RecipeChatScreen(
                uiState = uiState,
                uiEffect = uiEffect,
                onAction = viewModel::onAction,
                onNavigateToHome = {
                    navController.popBackStack()
                }
            )
        }
    }
}
