package com.example.my_fridge_android.ui.registe

import androidx.compose.ui.tooling.preview.PreviewParameterProvider

class RegisteScreenPreviewProvider : PreviewParameterProvider<RegisteContract.UiState> {

    override val values: Sequence<RegisteContract.UiState> = sequenceOf(
        RegisteContract.UiState(),
        RegisteContract.UiState(
            username = "johndoe",
            email = "john@example.com",
            password = "password123",
            confirmPassword = "password123",
            age = "25",
            height = "180",
            weight = "75",
            gender = "male",
            isRegisterEnabled = true
        ),
        RegisteContract.UiState(
            username = "jane",
            email = "invalid-email",
            password = "123",
            confirmPassword = "456",
            age = "abc",
            height = "50",
            weight = "10",
            gender = "female",
            isUsernameError = true,
            isEmailError = true,
            isPasswordError = true,
            isConfirmPasswordError = true,
            isAgeError = true,
            isHeightError = true,
            isWeightError = true,
            usernameErrorMessage = "Username must be at least 3 characters",
            emailErrorMessage = "Please enter a valid email",
            passwordErrorMessage = "Password must be at least 6 characters",
            confirmPasswordErrorMessage = "Passwords do not match",
            ageErrorMessage = "Age must be between 13 and 120",
            heightErrorMessage = "Height must be between 100 and 250 cm",
            weightErrorMessage = "Weight must be between 30 and 300 kg"
        ),
        RegisteContract.UiState(
            isLoading = true
        )
    )
}
