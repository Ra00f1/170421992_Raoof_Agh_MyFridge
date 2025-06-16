package com.example.my_fridge_android.ui.home

import androidx.compose.ui.tooling.preview.PreviewParameterProvider

class HomeScreenPreviewProvider : PreviewParameterProvider<HomeContract.UiState> {
    override val values: Sequence<HomeContract.UiState>
        get() = sequenceOf(
            HomeContract.UiState(
                isLoading = true,
                userName = "John Doe",
                userId = 123,
                isSendingToRecipeAssistant = false
            ),
            HomeContract.UiState(
                isLoading = false,
                userName = "John Doe",
                userId = 123,
                isSendingToRecipeAssistant = false
            ),
            HomeContract.UiState(
                isLoading = false,
                userName = "Alice Smith",
                userId = 456,
                isSendingToRecipeAssistant = true
            ),
        )
}
