package com.example.my_fridge_android.ui.registe

object RegisteContract {
    data class UiState(
        val isLoading: Boolean = false,
        val username: String = "",
        val email: String = "",
        val password: String = "",
        val confirmPassword: String = "",
        val age: String = "",
        val height: String = "",
        val weight: String = "",
        val gender: String = "male",
        val isPasswordVisible: Boolean = false,
        val isConfirmPasswordVisible: Boolean = false,
        val isUsernameError: Boolean = false,
        val isEmailError: Boolean = false,
        val isPasswordError: Boolean = false,
        val isConfirmPasswordError: Boolean = false,
        val isAgeError: Boolean = false,
        val isHeightError: Boolean = false,
        val isWeightError: Boolean = false,
        val usernameErrorMessage: String = "",
        val emailErrorMessage: String = "",
        val passwordErrorMessage: String = "",
        val confirmPasswordErrorMessage: String = "",
        val ageErrorMessage: String = "",
        val heightErrorMessage: String = "",
        val weightErrorMessage: String = "",
        val isRegisterEnabled: Boolean = false,
        val showGenderDropdown: Boolean = false,
    )

    sealed class UiAction {
        data class UsernameChanged(val username: String) : UiAction()
        data class EmailChanged(val email: String) : UiAction()
        data class PasswordChanged(val password: String) : UiAction()
        data class ConfirmPasswordChanged(val confirmPassword: String) : UiAction()
        data class AgeChanged(val age: String) : UiAction()
        data class HeightChanged(val height: String) : UiAction()
        data class WeightChanged(val weight: String) : UiAction()
        data class GenderSelected(val gender: String) : UiAction()
        data object TogglePasswordVisibility : UiAction()
        data object ToggleConfirmPasswordVisibility : UiAction()
        data object ToggleGenderDropdown : UiAction()
        data object RegisterClicked : UiAction()
        data object BackToLoginClicked : UiAction()
    }

    sealed class UiEffect {
        data object NavigateToLogin : UiEffect()
        data object NavigateToHome : UiEffect()
        data class ShowError(val message: String) : UiEffect()
    }
}
