from django import forms

class ChatForm(forms.Form):
    message = forms.CharField(
        label='',
        max_length=1000,
        widget=forms.TextInput(attrs={
            'id': 'prompt',
            'class': 'input',
            'placeholder': 'Type your message and press Enter...'
        })
    )
