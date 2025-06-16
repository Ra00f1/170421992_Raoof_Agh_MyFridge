package com.example.my_fridge_android.ui.registe

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalFocusManager
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
import androidx.compose.foundation.verticalScroll
import com.example.my_fridge_android.ui.components.LoadingBar
import com.example.my_fridge_android.ui.registe.RegisteContract.UiAction
import com.example.my_fridge_android.ui.registe.RegisteContract.UiEffect
import com.example.my_fridge_android.ui.registe.RegisteContract.UiState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisteScreen(
    uiState: UiState,
    uiEffect: Flow<UiEffect>,
    onAction: (UiAction) -> Unit,
    onNavigateToLogin: () -> Unit = {},
    onNavigateToHome: () -> Unit = {},
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
                is UiEffect.NavigateToLogin -> onNavigateToLogin()
                is UiEffect.NavigateToHome -> onNavigateToHome()
                is UiEffect.ShowError -> {
                    snackbarHostState.showSnackbar(
                        message = effect.message,
                        duration = SnackbarDuration.Long
                    )
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
                RegisteContent(
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisteContent(
    uiState: UiState,
    onAction: (UiAction) -> Unit
) {
    val scrollState = rememberScrollState()
    val focusManager = LocalFocusManager.current
    val emailFocusRequester = remember { FocusRequester() }
    val passwordFocusRequester = remember { FocusRequester() }
    val confirmPasswordFocusRequester = remember { FocusRequester() }
    val ageFocusRequester = remember { FocusRequester() }
    val heightFocusRequester = remember { FocusRequester() }
    val weightFocusRequester = remember { FocusRequester() }

    val genderOptions = listOf("male", "female", "prefer not to say")

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {
        // Top bar with back button
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp, bottom = 32.dp),
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = { onAction(UiAction.BackToLoginClicked) }
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "Back to Login",
                    tint = MaterialTheme.colorScheme.onSurface
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

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
            text = "Create Account",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = "Sign up to start managing your smart fridge",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        // Registration Form Card
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
                        onNext = { emailFocusRequester.requestFocus() }
                    ),
                    isError = uiState.isUsernameError,
                    supportingText = if (uiState.isUsernameError) {
                        { Text(uiState.usernameErrorMessage) }
                    } else null,
                    shape = RoundedCornerShape(16.dp)
                )

                // Email Field
                OutlinedTextField(
                    value = uiState.email,
                    onValueChange = { onAction(UiAction.EmailChanged(it)) },
                    label = { Text("Email") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Email,
                            contentDescription = "Email Icon"
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(emailFocusRequester),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Email,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { passwordFocusRequester.requestFocus() }
                    ),
                    isError = uiState.isEmailError,
                    supportingText = if (uiState.isEmailError) {
                        { Text(uiState.emailErrorMessage) }
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
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { confirmPasswordFocusRequester.requestFocus() }
                    ),
                    isError = uiState.isPasswordError,
                    supportingText = if (uiState.isPasswordError) {
                        { Text(uiState.passwordErrorMessage) }
                    } else null,
                    shape = RoundedCornerShape(16.dp)
                )

                // Confirm Password Field
                OutlinedTextField(
                    value = uiState.confirmPassword,
                    onValueChange = { onAction(UiAction.ConfirmPasswordChanged(it)) },
                    label = { Text("Confirm Password") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = "Confirm Password Icon"
                        )
                    },
                    trailingIcon = {
                        TextButton(
                            onClick = { onAction(UiAction.ToggleConfirmPasswordVisibility) }
                        ) {
                            Text(
                                text = if (uiState.isConfirmPasswordVisible) "Hide" else "Show",
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    },
                    visualTransformation = if (uiState.isConfirmPasswordVisible)
                        VisualTransformation.None
                    else
                        PasswordVisualTransformation(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(confirmPasswordFocusRequester),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Password,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { ageFocusRequester.requestFocus() }
                    ),
                    isError = uiState.isConfirmPasswordError,
                    supportingText = if (uiState.isConfirmPasswordError) {
                        { Text(uiState.confirmPasswordErrorMessage) }
                    } else null,
                    shape = RoundedCornerShape(16.dp)
                )

                // Age and Gender Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Age Field
                    OutlinedTextField(
                        value = uiState.age,
                        onValueChange = { onAction(UiAction.AgeChanged(it)) },
                        label = { Text("Age") },
                        modifier = Modifier
                            .weight(1f)
                            .focusRequester(ageFocusRequester),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Number,
                            imeAction = ImeAction.Next
                        ),
                        keyboardActions = KeyboardActions(
                            onNext = { heightFocusRequester.requestFocus() }
                        ),
                        isError = uiState.isAgeError,
                        supportingText = if (uiState.isAgeError) {
                            { Text(uiState.ageErrorMessage) }
                        } else null,
                        shape = RoundedCornerShape(16.dp)
                    )

                    // Gender Dropdown
                    ExposedDropdownMenuBox(
                        expanded = uiState.showGenderDropdown,
                        onExpandedChange = { onAction(UiAction.ToggleGenderDropdown) },
                        modifier = Modifier.weight(1f)
                    ) {
                        OutlinedTextField(
                            value = uiState.gender.replaceFirstChar { it.uppercase() },
                            onValueChange = { },
                            readOnly = true,
                            label = { Text("Gender") },
                            trailingIcon = {
                                Icon(
                                    imageVector = Icons.Default.ArrowDropDown,
                                    contentDescription = "Gender Dropdown"
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .menuAnchor(),
                            shape = RoundedCornerShape(16.dp)
                        )

                        ExposedDropdownMenu(
                            expanded = uiState.showGenderDropdown,
                            onDismissRequest = { onAction(UiAction.ToggleGenderDropdown) }
                        ) {
                            genderOptions.forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option.replaceFirstChar { it.uppercase() }) },
                                    onClick = { onAction(UiAction.GenderSelected(option)) }
                                )
                            }
                        }
                    }
                }

                // Height and Weight Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Height Field
                    OutlinedTextField(
                        value = uiState.height,
                        onValueChange = { onAction(UiAction.HeightChanged(it)) },
                        label = { Text("Height (cm)") },
                        modifier = Modifier
                            .weight(1f)
                            .focusRequester(heightFocusRequester),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Number,
                            imeAction = ImeAction.Next
                        ),
                        keyboardActions = KeyboardActions(
                            onNext = { weightFocusRequester.requestFocus() }
                        ),
                        isError = uiState.isHeightError,
                        supportingText = if (uiState.isHeightError) {
                            { Text(uiState.heightErrorMessage) }
                        } else null,
                        shape = RoundedCornerShape(16.dp)
                    )

                    // Weight Field
                    OutlinedTextField(
                        value = uiState.weight,
                        onValueChange = { onAction(UiAction.WeightChanged(it)) },
                        label = { Text("Weight (kg)") },
                        modifier = Modifier
                            .weight(1f)
                            .focusRequester(weightFocusRequester),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Number,
                            imeAction = ImeAction.Done
                        ),
                        keyboardActions = KeyboardActions(
                            onDone = {
                                focusManager.clearFocus()
                                if (uiState.isRegisterEnabled) {
                                    onAction(UiAction.RegisterClicked)
                                }
                            }
                        ),
                        isError = uiState.isWeightError,
                        supportingText = if (uiState.isWeightError) {
                            { Text(uiState.weightErrorMessage) }
                        } else null,
                        shape = RoundedCornerShape(16.dp)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Register Button
                Button(
                    onClick = {
                        println("DEBUG: Register button clicked! isRegisterEnabled=${uiState.isRegisterEnabled}")
                        onAction(UiAction.RegisterClicked)
                    },
                    enabled = uiState.isRegisterEnabled,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text(
                        text = if (uiState.isLoading) "Creating Account..." else "Create Account",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        // Back to Login Option
        Row(
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Already have an account? ",
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Sign In",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clickable { onAction(UiAction.BackToLoginClicked) }
            )
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Preview(showBackground = true)
@Composable
fun RegisteScreenPreview(
    @PreviewParameter(RegisteScreenPreviewProvider::class) uiState: UiState,
) {
    MaterialTheme {
        RegisteScreen(
            uiState = uiState,
            uiEffect = emptyFlow(),
            onAction = {},
        )
    }
}
