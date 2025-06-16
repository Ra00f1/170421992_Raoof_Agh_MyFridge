package com.example.my_fridge_android.ui.home

object HomeContract {
    data class UiState(
        val isLoading: Boolean = false,
        val userName: String = "User",
        val userId: Int = -1,
        val isSendingToRecipeAssistant: Boolean = false,
    )

    sealed class UiAction {
        object NavigateToReceiptList : UiAction()
        object NavigateToRecipeChat : UiAction()
        object SendItemsToRecipeAssistant : UiAction()
        object Logout : UiAction()
    }

    sealed class UiEffect {
        object NavigateToReceiptList : UiEffect()
        object NavigateToRecipeChat : UiEffect()
        object NavigateToLogin : UiEffect()
        data class ShowMessage(val message: String) : UiEffect()
        data class NavigateToRecipeChatWithResponse(val response: String) : UiEffect()
    }
}
