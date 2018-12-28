import React, {Component} from 'react';
import {Button} from 'reactstrap';
import logo from './logo.svg';
import './App.css';

class App extends Component {
    state = {
        msg: null
    };

    onClick = () => {
        this.setState({msg: 'You clicked the button, didn\'t you?'});
    };

    render() {
        const msg = this.state.msg ? this.state.msg : 'Hello, world!';

        return (
            <div className="App">
                <header className="App-header">
                    <img src={logo} className="App-logo" alt="logo"/>
                    <p>
                        {msg}
                    </p>
                    <Button color="danger" onClick={this.onClick}>Do not press!</Button>
                    <a
                        className="App-link"
                        href="https://reactjs.org"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Learn React
                    </a>
                </header>
            </div>
        );
    }
}

export default App;
