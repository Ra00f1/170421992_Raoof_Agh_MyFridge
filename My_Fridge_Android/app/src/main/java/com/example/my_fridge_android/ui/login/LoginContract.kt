package com.example.my_fridge_android.ui.login

object LoginContract {
    data class UiState(
        val isLoading: Boolean = false,
        val username: String = "",
        val password: String = "",
        val isUsernameError: Boolean = false,
        val isPasswordError: Boolean = false,
        val usernameErrorMessage: String = "",
        val passwordErrorMessage: String = "",
        val isPasswordVisible: Boolean = false,
        val isLoginEnabled: Boolean = false,
    )

    sealed class UiAction {
        data class UsernameChanged(val username: String) : UiAction()
        data class PasswordChanged(val password: String) : UiAction()
        object TogglePasswordVisibility : UiAction()
        object LoginClicked : UiAction()
        object ForgotPasswordClicked : UiAction()
        object SignUpClicked : UiAction()
    }

    sealed class UiEffect {
        object NavigateToHome : UiEffect()
        object NavigateToForgotPassword : UiEffect()
        object NavigateToSignUp : UiEffect()
        data class ShowError(val message: String) : UiEffect()
        data class ShowLoginResponse(val response: String) : UiEffect()
    }
}
