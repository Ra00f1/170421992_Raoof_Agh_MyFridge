package com.example.my_fridge_android.ui.login

import androidx.compose.ui.tooling.preview.PreviewParameterProvider

class LoginScreenPreviewProvider : PreviewParameterProvider<LoginContract.UiState> {
    override val values: Sequence<LoginContract.UiState>
        get() = sequenceOf(
            LoginContract.UiState(
                isLoading = false,
                username = "",
                password = "",
                isUsernameError = false,
                isPasswordError = false,
                isLoginEnabled = false
            ),
            LoginContract.UiState(
                isLoading = false,
                username = "testuser",
                password = "password123",
                isUsernameError = false,
                isPasswordError = false,
                isLoginEnabled = true
            ),
            LoginContract.UiState(
                isLoading = true,
                username = "",
                password = "",
                isUsernameError = false,
                isPasswordError = false,
                isLoginEnabled = false
            ),
        )
}
