package com.example.my_fridge_android.ui.login

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.my_fridge_android.R
import com.example.my_fridge_android.ui.components.LoadingBar
import com.example.my_fridge_android.ui.login.LoginContract.UiAction
import com.example.my_fridge_android.ui.login.LoginContract.UiEffect
import com.example.my_fridge_android.ui.login.LoginContract.UiState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

@Composable
fun LoginScreen(
    uiState: UiState,
    uiEffect: Flow<UiEffect>,
    onAction: (UiAction) -> Unit,
    onNavigateToHome: () -> Unit = {},
    onNavigateToSignUp: () -> Unit = {},
    onNavigateToForgotPassword: () -> Unit = {},
) {
    var isVisible by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(Unit) {
        delay(100)
        isVisible = true
    }

    // Handle UI effects
    LaunchedEffect(uiEffect) {
        uiEffect.collect { effect ->
            when (effect) {
                is UiEffect.NavigateToHome -> onNavigateToHome()
                is UiEffect.NavigateToSignUp -> onNavigateToSignUp()
                is UiEffect.NavigateToForgotPassword -> onNavigateToForgotPassword()
                is UiEffect.ShowError -> {
                    snackbarHostState.showSnackbar(
                        message = effect.message,
                        duration = SnackbarDuration.Long
                    )
                }
                is UiEffect.ShowLoginResponse -> {
                    // Just log the response, don't show popup
                    println("Login Response: ${effect.response}")
                }
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                        MaterialTheme.colorScheme.surface
                    )
                )
            )
    ) {
        if (uiState.isLoading) {
            LoadingBar()
        } else {
            AnimatedVisibility(
                visible = isVisible,
                enter = fadeIn(animationSpec = tween(800)) + slideInVertically(
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessLow
                    ),
                    initialOffsetY = { it / 2 }
                ),
                exit = fadeOut(animationSpec = tween(400)) + slideOutVertically()
            ) {
                LoginContent(
                    uiState = uiState,
                    onAction = onAction
                )
            }
        }

        // Snackbar for error messages
        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}

@Composable
fun LoginContent(
    uiState: UiState,
    onAction: (UiAction) -> Unit
) {
    val scrollState = rememberScrollState()
    val focusManager = LocalFocusManager.current
    val passwordFocusRequester = remember { FocusRequester() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Spacer(modifier = Modifier.height(32.dp))

        // App Logo/Icon
        Card(
            modifier = Modifier
                .size(120.dp)
                .padding(bottom = 32.dp),
            shape = RoundedCornerShape(60.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "🧊",
                    fontSize = 48.sp
                )
            }
        }

        // Welcome Text
        Text(
            text = "Welcome Back!",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = "Sign in to access your smart fridge",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        // Login Form Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Username Field
                OutlinedTextField(
                    value = uiState.username,
                    onValueChange = { onAction(UiAction.UsernameChanged(it)) },
                    label = { Text("Username") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Person,
                            contentDescription = "Username Icon"
                        )
                    },
                    modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Text,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { passwordFocusRequester.requestFocus() }
                    ),
                    isError = uiState.isUsernameError,
                    supportingText = if (uiState.isUsernameError) {
                        { Text(uiState.usernameErrorMessage) }
                    } else null,
                    shape = RoundedCornerShape(16.dp)
                )

                // Password Field
                OutlinedTextField(
                    value = uiState.password,
                    onValueChange = { onAction(UiAction.PasswordChanged(it)) },
                    label = { Text("Password") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = "Password Icon"
                        )
                    },
                    trailingIcon = {
                        TextButton(
                            onClick = { onAction(UiAction.TogglePasswordVisibility) }
                        ) {
                            Text(
                                text = if (uiState.isPasswordVisible) "Hide" else "Show",
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    },
                    visualTransformation = if (uiState.isPasswordVisible)
                        VisualTransformation.None
                    else
                        PasswordVisualTransformation(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(passwordFocusRequester),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Password,
                        imeAction = ImeAction.Done
                    ),
                    keyboardActions = KeyboardActions(
                        onDone = {
                            focusManager.clearFocus()
                            if (uiState.isLoginEnabled) {
                                onAction(UiAction.LoginClicked)
                            }
                        }
                    ),
                    isError = uiState.isPasswordError,
                    supportingText = if (uiState.isPasswordError) {
                        { Text(uiState.passwordErrorMessage) }
                    } else null,
                    shape = RoundedCornerShape(16.dp)
                )

                // Forgot Password
                Text(
                    text = "Forgot Password?",
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier
                        .align(Alignment.End)
                        .clickable { onAction(UiAction.ForgotPasswordClicked) }
                        .padding(vertical = 4.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Login Button
                Button(
                    onClick = {
                        println("DEBUG: Login button clicked! isLoginEnabled=${uiState.isLoginEnabled}")
                        onAction(UiAction.LoginClicked)
                    },
                    enabled = uiState.isLoginEnabled,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text(
                        text = if (uiState.isLoading) "Signing In..." else "Sign In",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        // Sign Up Option
        Row(
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Don't have an account? ",
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Sign Up",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clickable { onAction(UiAction.SignUpClicked) }
            )
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Preview(showBackground = true)
@Composable
fun LoginScreenPreview(
    @PreviewParameter(LoginScreenPreviewProvider::class) uiState: UiState,
) {
    MaterialTheme {
        LoginScreen(
            uiState = uiState,
            uiEffect = emptyFlow(),
            onAction = {},
        )
    }
}
